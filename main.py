import EvoSystem

EvoSystem.StartSim(
    AmountOfGen = 80,
    Seed = 1,
    UseLinReg = 1, # True for LinReg false for Sine
    UseCrowding = 0, # 0 for no, 1 for DeJong, And 2 for deterministic
    UseReplacementSelection = 0,
    UseFitnessSelection = 1,
    BitstringLength = 101, #101 for LinReg
    MutationRate = 1,
    CrossoverRate = 0.7,
    PopSize = 10,
    AmountOfParents = 4, #should be same as PopSize for deterministic crowding
    Constraint = [80,100]
)