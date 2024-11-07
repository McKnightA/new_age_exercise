import util
import exercise


exercise_name = "basicSquat"

util.record_exercise_video(exercise_name)
util.generate_reference_exercise(exercise_name, ["torso", "left leg", "right leg"])

ex1 = exercise.ReferenceExercise(exercise_name)

num_reps = 5
strictness = 70
util.count_reps_live(ex1, num_reps, strictness)

util.compare_to_reference(exercise_name, ex1)


# todo:
#  add more exercises
#  make routines a thing
#  .
#  app opens to list of routines (maybe personal/community favorites)
#  select one and it opens to a list of exercises with sets and reps or durations for each
#  select an exercise and it shows an animation of the reference exercise in 3d form
#  click a ready button and it swaps the animation out for a live stream with an overlay of the pose estimation
#  it counts the reps aloud(toggleable)
#  once all reps are complete or the set is ended by the user, the original reference 3d form returns
#  the user's 3d form appears next to or overlapped with the reference
#  if the routine's rep count is met, the set count drops by one for that exercise
#  .
#  be able to create a new routine or exercise
#  be able to share new exercises and routines with the community
#  .
#  potentially be able to add a figure to the 3d skeleton animation
#  get personalization options for the figure as rewards for completing routines
#  buy different avatar coaches like goku, rocky balboa, danny davito, etc to represent the reference exercises

