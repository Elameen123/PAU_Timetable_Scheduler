#!/usr/bin/env python3

"""
Test script to verify the improved differential evolution implementation
prevents student group clashes and missing classes from the very beginning.
"""

from differential_evolution import DifferentialEvolution
from input_data import input_data

def test_clash_free_initialization():
    """Test that all chromosomes in initial population are clash-free"""
    print("🧪 TESTING CLASH-FREE INITIALIZATION")
    print("="*50)
    
    # Create DE instance with small population for testing
    de = DifferentialEvolution(input_data, pop_size=5, F=0.4, CR=0.9)
    
    print(f"📊 Test Results:")
    print(f"   Total events that should be scheduled: {len(de.events_list)}")
    print(f"   Total student groups: {len(de.student_groups)}")
    print(f"   Population size: {len(de.population)}")
    
    all_good = True
    
    for i, chromosome in enumerate(de.population):
        print(f"\n🔍 Checking Chromosome {i+1}:")
        
        # Check 1: All events scheduled
        scheduled_events = set()
        for room_idx in range(len(de.rooms)):
            for timeslot_idx in range(len(de.timeslots)):
                event_id = chromosome[room_idx][timeslot_idx]
                if event_id is not None:
                    scheduled_events.add(event_id)
        
        missing_count = len(de.events_list) - len(scheduled_events)
        print(f"   Events: {len(scheduled_events)}/{len(de.events_list)} scheduled, {missing_count} missing")
        
        if missing_count == 0:
            print("   ✅ All events scheduled")
        else:
            print(f"   ❌ {missing_count} events missing")
            all_good = False
        
        # Check 2: No student group clashes
        clash_count = 0
        for t_idx in range(len(de.timeslots)):
            student_groups_at_slot = set()
            for r_idx in range(len(de.rooms)):
                event_id = chromosome[r_idx, t_idx]
                if event_id is not None:
                    event = de.events_map.get(event_id)
                    if event:
                        sg_id = event.student_group.id
                        if sg_id in student_groups_at_slot:
                            clash_count += 1
                        else:
                            student_groups_at_slot.add(sg_id)
        
        if clash_count == 0:
            print("   ✅ No student group clashes")
        else:
            print(f"   ❌ {clash_count} student group clashes")
            all_good = False
        
        # Check 3: Basic fitness
        fitness = de.evaluate_fitness(chromosome)
        print(f"   Fitness: {fitness:.2f}")
    
    print("\n" + "="*50)
    if all_good:
        print("🎉 SUCCESS: All chromosomes are clash-free with no missing events!")
        return True
    else:
        print("❌ FAILURE: Some chromosomes have issues")
        return False

def test_quick_evolution():
    """Test a few generations to ensure properties are maintained"""
    print("\n🧪 TESTING SHORT EVOLUTION MAINTAINS PROPERTIES")
    print("="*50)
    
    # Create DE instance and run for a few generations
    de = DifferentialEvolution(input_data, pop_size=3, F=0.4, CR=0.9)
    
    print("Running 3 generations...")
    best_solution, fitness_history, generation, diversity_history = de.run(3)
    
    # Check final solution
    print(f"\n🔍 Checking Final Solution:")
    
    # Count scheduled events
    scheduled_events = set()
    for room_idx in range(len(de.rooms)):
        for timeslot_idx in range(len(de.timeslots)):
            event_id = best_solution[room_idx][timeslot_idx]
            if event_id is not None:
                scheduled_events.add(event_id)
    
    missing_count = len(de.events_list) - len(scheduled_events)
    print(f"   Events: {len(scheduled_events)}/{len(de.events_list)} scheduled, {missing_count} missing")
    
    # Count clashes
    clash_count = de.count_student_group_clashes(best_solution)
    print(f"   Student group clashes: {clash_count}")
    
    # Final fitness
    final_fitness = de.evaluate_fitness(best_solution)
    print(f"   Final fitness: {final_fitness:.2f}")
    
    success = (missing_count == 0 and clash_count == 0)
    if success:
        print("   ✅ Evolution maintained clash-free property!")
    else:
        print("   ❌ Evolution introduced problems")
    
    return success

if __name__ == "__main__":
    print("🚀 COMPREHENSIVE TESTING OF IMPROVED DIFFERENTIAL EVOLUTION")
    print("="*60)
    
    test1_passed = test_clash_free_initialization()
    test2_passed = test_quick_evolution()
    
    print("\n" + "="*60)
    print("📋 FINAL TEST SUMMARY:")
    print(f"   Clash-free initialization: {'✅ PASS' if test1_passed else '❌ FAIL'}")
    print(f"   Property maintenance:       {'✅ PASS' if test2_passed else '❌ FAIL'}")
    
    if test1_passed and test2_passed:
        print("\n🎉 ALL TESTS PASSED! The implementation is working correctly.")
        print("   Student group clashes and missing classes should now be ELIMINATED.")
    else:
        print("\n⚠️  SOME TESTS FAILED. Further debugging needed.")
