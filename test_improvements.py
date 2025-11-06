#!/usr/bin/env python3

from differential_evolution import DifferentialEvolution
from input_data import input_data

def test_improvements():
    print("Testing improved differential evolution algorithm...")
    print("=" * 60)
    
    # Create DE instance with smaller population for testing
    de = DifferentialEvolution(input_data, 10, 0.4, 0.9)
    
    print("\nEngineering groups identified:")
    for group_id in de.engineering_groups:
        group = next(g for g in de.student_groups if g.id == group_id)
        print(f"  {group_id}: {group.name}")
    
    print("\nNon-engineering groups:")
    for group in de.student_groups:
        if group.id not in de.engineering_groups:
            print(f"  {group.id}: {group.name}")
    
    print("\nRoom building assignments:")
    sst_rooms = []
    tyd_rooms = []
    for idx, building in de.room_building_cache.items():
        room = de.rooms[idx]
        room_name = getattr(room, 'name', f'Room {idx}')
        if building == 'SST':
            sst_rooms.append(f"{room_name} ({room.room_type})")
        elif building == 'TYD':
            tyd_rooms.append(f"{room_name} ({room.room_type})")
    
    print(f"  SST rooms ({len(sst_rooms)}): {', '.join(sst_rooms[:5])}{'...' if len(sst_rooms) > 5 else ''}")
    print(f"  TYD rooms ({len(tyd_rooms)}): {', '.join(tyd_rooms[:5])}{'...' if len(tyd_rooms) > 5 else ''}")
    
    # Test initial population
    print("\nTesting initial population...")
    initial_fitness = [de.evaluate_fitness(ind) for ind in de.population]
    best_fitness = min(initial_fitness)
    avg_fitness = sum(initial_fitness) / len(initial_fitness)
    
    print(f"Initial population results:")
    print(f"  Best fitness: {best_fitness:.1f}")
    print(f"  Average fitness: {avg_fitness:.1f}")
    
    # Show constraint breakdown for best initial solution
    best_idx = initial_fitness.index(best_fitness)
    best_solution = de.population[best_idx]
    violations = de.constraints.get_constraint_violations(best_solution)
    
    print(f"\nConstraint breakdown for best initial solution:")
    for constraint, value in violations.items():
        if value > 0:
            print(f"  {constraint}: {value:.1f}")
    
    # Test a few generations
    print(f"\nRunning 5 generations...")
    best_solution, fitness_history, generation, diversity_history = de.run(5)
    
    print(f"\nFinal results:")
    print(f"  Final best fitness: {fitness_history[-1]:.1f}")
    print(f"  Improvement: {best_fitness:.1f} → {fitness_history[-1]:.1f}")
    print(f"  Generations completed: {generation + 1}")

if __name__ == "__main__":
    test_improvements()
