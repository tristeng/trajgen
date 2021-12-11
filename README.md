# trajgen

**trajgen** is a minimum trajectory generation library written for Rust - given a set of 3D points,
desired times, and type of minimum trajectory, the library will calculate the appropriate order
polynomials to ensure a smooth path (as well as smooth derivatives) through the points.

## Features
* minimum trajectory generation for velocity, acceleration, jerk and snap
* calculate desired position, velocity, acceleration and further derivatives given time

## Using **trajgen**
Simply add the following to your `Cargo.toml` file:

```ignore
[dependencies]
trajgen = "*"
```

and now you can generate and use trajectories:

```
use trajgen::{TrajectoryGenerator, TrajectoryType, Point};

fn main() {
    let mut traj_gen = TrajectoryGenerator::new(TrajectoryType::Snap);

    // add waypoints for the trajectory
    let waypoints: Vec<Point> = vec![Point {x: 1., y: 2., z: 3., t: 0.},
                                     Point {x: 2., y: 3., z: 4., t: 1.},
                                     Point {x: 3., y: 4., z: 5., t: 2.}];
    // solve for given waypoints
    traj_gen.generate(&waypoints);

    // use the individual values in real-time, perhaps to control a robot
    let t = 0.24;
    let pos = traj_gen.get_position(t).unwrap();
    print!("Current desired position for time {} is {}", t, pos);

    let vel = traj_gen.get_velocity(t).unwrap();
    print!("Current desired velocity for time {} is {}", t, vel);

    let acc = traj_gen.get_acceleration(t).unwrap();
    print!("Current desired acceleration for time {} is {}", t, acc);

    // or get values for a range of times, perhaps to plot
    let path = traj_gen.get_positions(0.25, 0.75, 0.1);
}
```

## Derivation
For the derivation of the snap trajectories this library implements, please see 
the following [Jupyter notebook](https://github.com/tristeng/control/blob/master/notebooks/trajector-generator.ipynb).