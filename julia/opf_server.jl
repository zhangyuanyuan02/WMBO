#!/usr/bin/env julia

using JSON3
using Ipopt
using JuMP
using Logging
using PowerModels

const PROJECT_ROOT = normpath(joinpath(@__DIR__, ".."))
const CASE_PATHS = Dict(
    "opf_pglib_case14_typ_pg" => joinpath(
        PROJECT_ROOT,
        "data",
        "pglib-opf",
        "v23.07",
        "pglib_opf_case14_ieee.m",
    ),
    "opf_pglib_case14_api_pg" => joinpath(
        PROJECT_ROOT,
        "data",
        "pglib-opf",
        "v23.07",
        "api",
        "pglib_opf_case14_ieee__api.m",
    ),
)
const CASE_DATA = Dict{String, Dict{String, Any}}()
const CASE_DESCRIPTIONS = Dict{String, Dict{String, Any}}()
const REFERENCE_RESULTS = Dict{String, Dict{String, Any}}()
const SOLVER = JuMP.optimizer_with_attributes(
    Ipopt.Optimizer,
    "print_level" => 0,
    "sb" => "yes",
    "tol" => 1.0e-8,
    "max_iter" => 500,
)

global_logger(ConsoleLogger(stderr, Logging.Error))
PowerModels.logger_config!("error")

function _component_keys(data::Dict{String, Any}, component::String)
    return sort(collect(keys(data[component])); by = key -> parse(Int, key))
end

function _reference_bus_ids(data::Dict{String, Any})
    return Set(
        string(bus["index"])
        for bus in values(data["bus"])
        if Int(bus["bus_type"]) == 3
    )
end

function _describe_case(data::Dict{String, Any}, benchmark::String)
    base_mva = Float64(data["baseMVA"])
    reference_buses = _reference_bus_ids(data)
    controllable_ids = String[]
    bounds_mw = Vector{Vector{Float64}}()
    starts_mw = Float64[]
    slack_generator_ids = String[]

    for generator_id in _component_keys(data, "gen")
        generator = data["gen"][generator_id]
        Int(generator["gen_status"]) == 0 && continue
        generator_bus = string(Int(generator["gen_bus"]))
        if generator_bus in reference_buses
            push!(slack_generator_ids, generator_id)
            continue
        end
        pmin = Float64(generator["pmin"])
        pmax = Float64(generator["pmax"])
        if pmax - pmin > 1.0e-10
            push!(controllable_ids, generator_id)
            push!(bounds_mw, [pmin * base_mva, pmax * base_mva])
            push!(starts_mw, Float64(generator["pg"]) * base_mva)
        end
    end

    isempty(controllable_ids) && error("case has no non-slack active-power controls")
    return Dict{String, Any}(
        "benchmark" => benchmark,
        "scenario_id" => splitext(basename(CASE_PATHS[benchmark]))[1],
        "dim" => length(controllable_ids),
        "base_mva" => base_mva,
        "controllable_generator_ids" => controllable_ids,
        "slack_generator_ids" => slack_generator_ids,
        "bounds_mw" => bounds_mw,
        "recommended_start_mw" => starts_mw,
    )
end

function _load_cases!()
    for (benchmark, path) in CASE_PATHS
        isfile(path) || error("missing PGLib case file: $(path)")
        data = redirect_stdout(stderr) do
            PowerModels.parse_file(path)
        end
        CASE_DATA[benchmark] = data
        CASE_DESCRIPTIONS[benchmark] = _describe_case(data, benchmark)
    end
end

function _is_solved(result::Dict{String, Any})
    status = uppercase(string(get(result, "termination_status", "")))
    return status in ("LOCALLY_SOLVED", "ALMOST_LOCALLY_SOLVED", "OPTIMAL", "ALMOST_OPTIMAL")
end

function _restricted_reference(benchmark::String)
    if haskey(REFERENCE_RESULTS, benchmark)
        return REFERENCE_RESULTS[benchmark]
    end

    initial_data = deepcopy(CASE_DATA[benchmark])
    initial_result = PowerModels.solve_ac_opf(initial_data, SOLVER)
    _is_solved(initial_result) || error(
        "initial AC-OPF failed for $(benchmark): $(get(initial_result, "termination_status", "unknown"))",
    )
    voltage_setpoints = Dict{String, Float64}()
    for generator in values(initial_data["gen"])
        Int(generator["gen_status"]) == 0 && continue
        bus_id = string(Int(generator["gen_bus"]))
        voltage_setpoints[bus_id] = Float64(initial_result["solution"]["bus"][bus_id]["vm"])
    end

    data = deepcopy(CASE_DATA[benchmark])
    for (bus_id, setpoint) in voltage_setpoints
        bus = data["bus"][bus_id]
        bus["vm"] = setpoint
        bus["vmin"] = setpoint
        bus["vmax"] = setpoint
    end

    started = time()
    result = PowerModels.solve_ac_opf(data, SOLVER)
    _is_solved(result) || error(
        "restricted AC-OPF failed for $(benchmark): $(get(result, "termination_status", "unknown"))",
    )
    base_mva = Float64(data["baseMVA"])
    description = CASE_DESCRIPTIONS[benchmark]
    bounds_mw = description["bounds_mw"]
    pg_mw = [
        clamp(
            Float64(result["solution"]["gen"][generator_id]["pg"]) * base_mva,
            Float64(bounds_mw[index][1]),
            Float64(bounds_mw[index][2]),
        ) for (index, generator_id) in enumerate(description["controllable_generator_ids"])
    ]
    reference = Dict{String, Any}(
        "benchmark" => benchmark,
        "scenario_id" => description["scenario_id"],
        "reference_cost" => Float64(result["objective"]),
        "reference_pg_mw" => pg_mw,
        "voltage_setpoints" => voltage_setpoints,
        "reference_kind" => "restricted_pg_with_frozen_ac_opf_voltage_setpoints",
        "termination_status" => string(result["termination_status"]),
        "solve_time" => Float64(get(result, "solve_time", time() - started)),
    )
    REFERENCE_RESULTS[benchmark] = reference
    return reference
end

function _normalised_bound_excess(value::Float64, lower::Float64, upper::Float64)
    scale = max(abs(lower), abs(upper), upper - lower, 1.0e-8)
    if value < lower
        return (lower - value) / scale
    elseif value > upper
        return (value - upper) / scale
    end
    return 0.0
end

function _polynomial_cost(coefficients, value::Float64)
    result = 0.0
    for coefficient in coefficients
        result = result * value + Float64(coefficient)
    end
    return result
end

function _generation_cost(data::Dict{String, Any})
    total = 0.0
    for generator in values(data["gen"])
        Int(generator["gen_status"]) == 0 && continue
        Int(get(generator, "model", 2)) == 2 || error("only polynomial generator costs are supported")
        total += _polynomial_cost(generator["cost"], Float64(generator["pg"]))
    end
    return total
end

function _power_balance_residuals(data::Dict{String, Any})
    active_residuals = Float64[]
    reactive_residuals = Float64[]

    for bus_id in _component_keys(data, "bus")
        bus = data["bus"][bus_id]
        bus_index = Int(bus["index"])
        vm = Float64(bus["vm"])
        generated_p = sum(
            Float64(generator["pg"])
            for generator in values(data["gen"])
            if Int(generator["gen_status"]) != 0 && Int(generator["gen_bus"]) == bus_index;
            init = 0.0,
        )
        generated_q = sum(
            Float64(generator["qg"])
            for generator in values(data["gen"])
            if Int(generator["gen_status"]) != 0 && Int(generator["gen_bus"]) == bus_index;
            init = 0.0,
        )
        load_p = sum(
            Float64(load["pd"])
            for load in values(data["load"])
            if Int(load["status"]) != 0 && Int(load["load_bus"]) == bus_index;
            init = 0.0,
        )
        load_q = sum(
            Float64(load["qd"])
            for load in values(data["load"])
            if Int(load["status"]) != 0 && Int(load["load_bus"]) == bus_index;
            init = 0.0,
        )
        shunt_g = sum(
            Float64(shunt["gs"])
            for shunt in values(data["shunt"])
            if Int(shunt["status"]) != 0 && Int(shunt["shunt_bus"]) == bus_index;
            init = 0.0,
        )
        shunt_b = sum(
            Float64(shunt["bs"])
            for shunt in values(data["shunt"])
            if Int(shunt["status"]) != 0 && Int(shunt["shunt_bus"]) == bus_index;
            init = 0.0,
        )
        outgoing_p = 0.0
        outgoing_q = 0.0
        for branch in values(data["branch"])
            Int(branch["br_status"]) == 0 && continue
            if Int(branch["f_bus"]) == bus_index
                outgoing_p += Float64(branch["pf"])
                outgoing_q += Float64(branch["qf"])
            elseif Int(branch["t_bus"]) == bus_index
                outgoing_p += Float64(branch["pt"])
                outgoing_q += Float64(branch["qt"])
            end
        end
        push!(active_residuals, abs(generated_p - load_p - shunt_g * vm^2 - outgoing_p))
        push!(reactive_residuals, abs(generated_q - load_q + shunt_b * vm^2 - outgoing_q))
    end
    return active_residuals, reactive_residuals
end

function _constraint_diagnostics(data::Dict{String, Any})
    voltage_excesses = Float64[]
    thermal_excesses = Float64[]
    angle_excesses = Float64[]
    generator_excesses = Float64[]

    for bus in values(data["bus"])
        Int(bus["bus_type"]) == 4 && continue
        push!(
            voltage_excesses,
            _normalised_bound_excess(
                Float64(bus["vm"]),
                Float64(bus["vmin"]),
                Float64(bus["vmax"]),
            ),
        )
    end

    for generator in values(data["gen"])
        Int(generator["gen_status"]) == 0 && continue
        push!(
            generator_excesses,
            _normalised_bound_excess(
                Float64(generator["pg"]),
                Float64(generator["pmin"]),
                Float64(generator["pmax"]),
            ),
        )
        push!(
            generator_excesses,
            _normalised_bound_excess(
                Float64(generator["qg"]),
                Float64(generator["qmin"]),
                Float64(generator["qmax"]),
            ),
        )
    end

    for branch in values(data["branch"])
        Int(branch["br_status"]) == 0 && continue
        if haskey(branch, "rate_a") && Float64(branch["rate_a"]) > 0.0
            rate = Float64(branch["rate_a"])
            from_flow = hypot(Float64(branch["pf"]), Float64(branch["qf"]))
            to_flow = hypot(Float64(branch["pt"]), Float64(branch["qt"]))
            push!(thermal_excesses, max(0.0, from_flow - rate) / rate)
            push!(thermal_excesses, max(0.0, to_flow - rate) / rate)
        end
        from_bus = data["bus"][string(Int(branch["f_bus"]))]
        to_bus = data["bus"][string(Int(branch["t_bus"]))]
        angle_difference = Float64(from_bus["va"]) - Float64(to_bus["va"])
        push!(
            angle_excesses,
            _normalised_bound_excess(
                angle_difference,
                Float64(branch["angmin"]),
                Float64(branch["angmax"]),
            ),
        )
    end

    active_residuals, reactive_residuals = _power_balance_residuals(data)
    balance_residuals = vcat(active_residuals, reactive_residuals)
    all_excesses = vcat(
        voltage_excesses,
        thermal_excesses,
        angle_excesses,
        generator_excesses,
        balance_residuals,
    )
    return Dict{String, Any}(
        "total_violation" => sum(value^2 for value in all_excesses; init = 0.0),
        "max_normalized_violation" => maximum(all_excesses; init = 0.0),
        "max_voltage_violation" => maximum(voltage_excesses; init = 0.0),
        "max_thermal_violation" => maximum(thermal_excesses; init = 0.0),
        "max_angle_violation" => maximum(angle_excesses; init = 0.0),
        "generator_violation" => sum(value^2 for value in generator_excesses; init = 0.0),
        "power_balance_residual" => maximum(balance_residuals; init = 0.0),
    )
end

function _failed_evaluation(
    benchmark::String,
    reference_cost::Float64,
    elapsed::Float64,
    status::String,
    message::String,
)
    return Dict{String, Any}(
        "benchmark" => benchmark,
        "scenario_id" => CASE_DESCRIPTIONS[benchmark]["scenario_id"],
        "pf_converged" => false,
        "feasible" => false,
        "generation_cost" => nothing,
        "reference_cost" => reference_cost,
        "total_violation" => 1.0,
        "max_normalized_violation" => 1.0,
        "max_voltage_violation" => 0.0,
        "max_thermal_violation" => 0.0,
        "max_angle_violation" => 0.0,
        "generator_violation" => 0.0,
        "power_balance_residual" => 1.0,
        "evaluation_time" => elapsed,
        "termination_status" => status,
        "solver_error" => message,
    )
end

function _evaluate_case(benchmark::String, pg_mw, feasibility_tolerance::Float64)
    haskey(CASE_DATA, benchmark) || error("unknown OPF benchmark: $(benchmark)")
    description = CASE_DESCRIPTIONS[benchmark]
    controls = description["controllable_generator_ids"]
    length(pg_mw) == length(controls) || error(
        "expected $(length(controls)) Pg controls, got $(length(pg_mw))",
    )
    reference = _restricted_reference(benchmark)
    started = time()
    data = deepcopy(CASE_DATA[benchmark])
    base_mva = Float64(data["baseMVA"])
    for (bus_id, setpoint) in reference["voltage_setpoints"]
        data["bus"][bus_id]["vm"] = Float64(setpoint)
    end
    for (generator_id, value) in zip(controls, pg_mw)
        data["gen"][generator_id]["pg"] = Float64(value) / base_mva
    end

    result = try
        PowerModels.solve_ac_pf(data, SOLVER)
    catch error
        return _failed_evaluation(
            benchmark,
            Float64(reference["reference_cost"]),
            time() - started,
            "EXCEPTION",
            sprint(showerror, error),
        )
    end
    status = string(get(result, "termination_status", "unknown"))
    if !_is_solved(result)
        return _failed_evaluation(
            benchmark,
            Float64(reference["reference_cost"]),
            time() - started,
            status,
            "AC power flow did not converge",
        )
    end

    PowerModels.update_data!(data, result["solution"])
    PowerModels.update_data!(data, PowerModels.calc_branch_flow_ac(data))
    diagnostics = _constraint_diagnostics(data)
    generation_cost = _generation_cost(data)
    max_violation = Float64(diagnostics["max_normalized_violation"])
    return merge(
        diagnostics,
        Dict{String, Any}(
            "benchmark" => benchmark,
            "scenario_id" => description["scenario_id"],
            "pf_converged" => true,
            "feasible" => max_violation <= feasibility_tolerance,
            "generation_cost" => generation_cost,
            "reference_cost" => Float64(reference["reference_cost"]),
            "evaluation_time" => time() - started,
            "reference_kind" => reference["reference_kind"],
            "solver_time" => Float64(get(result, "solve_time", 0.0)),
            "termination_status" => status,
        ),
    )
end

function _request_value(request::Dict{String, Any}, key::String, default)
    return haskey(request, key) ? request[key] : default
end

function _handle_request(request::Dict{String, Any})
    action = lowercase(string(_request_value(request, "action", "")))
    if action == "describe"
        benchmark = string(request["benchmark"])
        haskey(CASE_DESCRIPTIONS, benchmark) || error("unknown OPF benchmark: $(benchmark)")
        return CASE_DESCRIPTIONS[benchmark]
    elseif action == "reference"
        return _restricted_reference(string(request["benchmark"]))
    elseif action == "evaluate"
        benchmark = string(request["benchmark"])
        pg_mw = Float64.(request["pg_mw"])
        tolerance = Float64(_request_value(request, "feasibility_tolerance", 1.0e-5))
        return _evaluate_case(benchmark, pg_mw, tolerance)
    elseif action == "warmup"
        warmed = Dict{String, Any}()
        for benchmark in sort(collect(keys(CASE_DATA)))
            reference = _restricted_reference(benchmark)
            start = reference["reference_pg_mw"]
            evaluation = _evaluate_case(benchmark, start, 1.0e-5)
            warmed[benchmark] = Dict(
                "reference_cost" => reference["reference_cost"],
                "base_pf_converged" => evaluation["pf_converged"],
            )
        end
        return Dict("warmed" => warmed)
    elseif action == "shutdown"
        return Dict("shutdown" => true)
    end
    error("unknown action: $(action)")
end

function _serve()
    _load_cases!()
    for line in eachline(stdin)
        isempty(strip(line)) && continue
        request_id = nothing
        response = try
            request = JSON3.read(line, Dict{String, Any})
            request_id = get(request, "request_id", nothing)
            Dict{String, Any}(
                "ok" => true,
                "request_id" => request_id,
                "result" => _handle_request(request),
            )
        catch error
            Dict{String, Any}(
                "ok" => false,
                "request_id" => request_id,
                "error" => sprint(showerror, error, catch_backtrace()),
            )
        end
        println(stdout, JSON3.write(response))
        flush(stdout)
        if get(response, "ok", false) && haskey(response, "result")
            result = response["result"]
            if result isa Dict && get(result, "shutdown", false)
                break
            end
        end
    end
end

_serve()
