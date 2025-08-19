def mututate(self, chromosome):
        mnutation_rate = 0.1

        # Traverse through the chromosome
        for room_idx in range(chromosome.shape[0]):
            for timeslot_idx in range(chromosome.shape[1]):
                # Apply mutation with a certain probability
                if random.random() < mnutation_rate:
                    event = chromosome[room_idx][timeslot_idx]

                    # If there's no event assgined, skip mutation
                    if event is None:
                        continue

                    # Randomly choose the type of mutation
                    mutation_type = random.choice(['swap', 'change_room', 'chane_timeslot'])

                    if mutation_type == 'swap':
                        # Find another random event to swap with
                        new_room_idx = random.randint(0, chromosome.shape[0] - 1)
                        new_timeslot_idx = random.randint(0, chromosome.shape[1] - 1)

                        # Swap the events
                        chromosome[room_idx][timeslot_idx], chromosome[new_room_idx][new_timeslot_idx] = (chromosome[new_room_idx][new_timeslot_idx], chromosome[room_idx][timeslot_idx])

                    
                    elif mutation_type == 'change_room':
                        # Find a new random room for this event
                        new_room_idx = random.randint(0, chromosome.shape[0] - 1)

                        if self.is_room_suitable(self.rooms[new_room_idx], input_data.getCourse(event.course_id)):
                            # Move the event to the new room (if available)
                            chromosome[room_idx][timeslot_idx], chromosome[new_room_idx][timeslot_idx] = (chromosome[new_room_idx][timeslot_idx], chromosome[room_idx][timeslot_idx])

                    
                    elif mutation_type == 'change_timeslot':
                        # Find a new random room for this event
                        new_timeslot_idx = random.randint(0, chromosome.shape[1] - 1)

                        if self.is_slot_available(room_idx, new_timeslot_idx):
                            # Move the event to the new room (if available)
                            chromosome[room_idx][timeslot_idx], chromosome[new_room_idx][new_timeslot_idx] = (chromosome[room_idx][new_timeslot_idx], chromosome[room_idx][timeslot_idx])
