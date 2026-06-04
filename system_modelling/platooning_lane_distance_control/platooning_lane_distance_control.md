
# Scenario 2: Platooning/Lane and Distance Control

## Short description

The Platooning/Lane and Distance Control scenario describes the steady-state platooning behavior after a truck has 
already joined the platoon. In this scenario, the following truck receives control messages, measures the distance to 
the preceding truck, detects the lane/track, and updates the trajectory value needed to follow the preceding truck.

The following truck regulates its motion to maintain at least the minimum safe distance and blocks acceleration if the 
distance becomes unsafe or lane/track detection is lost. If an emergency braking message is detected, steady-state 
platooning stops and the truck performs emergency braking.

## Functional and non-functional requirements

| ID      | Type       | Requirement                                                                                                                                                  |
|---------|------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------|
| S2-F-01 | Functional | Each following truck shall receive a control message from at least the preceding truck.                                                                      |
| S2-F-02 | Functional | Each following truck shall measure the distance to the preceding truck.                                                                                      |
| S2-F-03 | Functional | Each following truck shall regulate motion to maintain at least the minimum safe distance.                                                                   |
| S2-F-04 | Functional | Each following truck shall detect the lane/track.                                                                                                            |
| S2-F-05 | Functional | The system shall update a trajectory value for the following truck to follow the preceding truck.                                                            |
| S2-F-06 | Functional | The following truck shall perform emergency braking when an emergency braking message is detected.                                                           |
| S2-T-01 | Timing     | The system shall update its control state every 100 ms.                                                                                                      |
| S2-T-02 | Timing     | If the distance becomes less than the minimum safe distance, the following truck shall start reducing speed within 100 ms to restore safe distance.          |
| S2-T-03 | Timing     | Lane/track detection and trajectory value update shall finish within 100 ms.                                                                                 |
| S2-S-01 | Safety     | Steady-state platooning shall not continue when an emergency braking message is detected.                                                                    |
| S2-S-02 | Safety     | A following truck shall block acceleration when the distance to the preceding truck is below the minimum safe distance or when lane/track detection is lost. |
| S2-S-03 | Safety     | The system shall never allow the distance to the preceding truck to be below the minimum safe distance.                                                      |
