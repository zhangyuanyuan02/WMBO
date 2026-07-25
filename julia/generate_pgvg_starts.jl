#!/usr/bin/env julia

include(joinpath(@__DIR__, "opf_server.jl"))

function _fixed_pg_candidate(benchmark::String, target_mw::Float64)
    description = CASE_DESCRIPTIONS[benchmark]
    generator_ids = description["controllable_generator_ids"]
    length(generator_ids) == 1 || error("start generation expects one Pg control")
    data = deepcopy(CASE_DATA[benchmark])
    generator = data["gen"][generator_ids[1]]
    target_pu = target_mw / Float64(data["baseMVA"])
    generator["pmin"] = target_pu
    generator["pmax"] = target_pu
    result = PowerModels.solve_ac_opf(data, SOLVER)
    _is_solved(result) || return nothing
    controls = _extract_control_values(description, result["solution"])
    evaluation = _evaluate_case(benchmark, controls, 1.0e-5)
    Bool(evaluation["pf_converged"]) || return nothing
    Bool(evaluation["feasible"]) || return nothing
    reference_cost = Float64(evaluation["reference_cost"])
    gap = max(
        0.0,
        (Float64(evaluation["generation_cost"]) - reference_cost) /
        max(abs(reference_cost), 1.0),
    )
    return Dict(
        "recommended_start" => controls,
        "generation_cost" => evaluation["generation_cost"],
        "reference_cost" => reference_cost,
        "normalised_cost_gap" => gap,
        "total_violation" => evaluation["total_violation"],
    )
end

function _find_start(benchmark::String)
    description = CASE_DESCRIPTIONS[benchmark]
    pg_variable = first(variable for variable in description["variables"] if variable["kind"] == "pg")
    lower = Float64(pg_variable["lower"])
    upper = Float64(pg_variable["upper"])
    candidates = Dict{String, Any}[]
    for target in range(lower, upper; length = 101)
        candidate = _fixed_pg_candidate(benchmark, Float64(target))
        candidate === nothing || push!(candidates, candidate)
    end
    eligible = [
        candidate for candidate in candidates
        if Float64(candidate["normalised_cost_gap"]) >= 0.01
    ]
    isempty(eligible) && error("no strictly feasible start with at least 1% cost gap for $(benchmark)")
    sort!(eligible; by = candidate -> Float64(candidate["normalised_cost_gap"]))
    return eligible[end]
end

_load_cases!()
starts = Dict(benchmark => _find_start(benchmark) for benchmark in sort(collect(keys(CASE_DATA))))
println(JSON3.write(starts))
