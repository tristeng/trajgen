/*!
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
For the derivation of the minimum trajectories this library implements, please see
the following [Jupyter notebook](https://github.com/tristeng/control/blob/master/notebooks/trajector-generator.ipynb).

 */

use log::{info};
use std::fmt;
use std::cmp::Ordering;

use nalgebra::{Matrix, Dyn, VecStorage, U1, Vector, RowVector};
use std::fmt::Formatter;

// Define the dynamic matrix we use to solve
type DMatrixf32 = Matrix<f32, Dyn, Dyn, VecStorage<f32, Dyn, Dyn>>;
type DColVectorf32 = Vector<f32, Dyn, VecStorage<f32, Dyn, U1>>;
type DRowVectorf32 = RowVector<f32, Dyn, VecStorage<f32, U1, Dyn>>;

/// a point in 3D space-time
#[derive(Debug, Copy, Clone)]
pub struct Point {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub t: f32,
}

impl Point {
    /// Creates a point in 3D space-time
    /// 
    /// # Arguments
    /// 
    /// `x` - value along the x-axis
    /// `y` - value along the y-axis
    /// `z` - value along the z-axis
    /// `t` - value along the time axis
    ///
    /// # Examples
    /// ```
    /// use trajgen::Point;
    /// let p3d = Point::new(5.2, -6.75, 0.23, 1.43);
    /// ```
    pub fn new(x: f32, y: f32, z: f32, t: f32) -> Point {
        Point { x, y, z, t }
    }
}

impl fmt::Display for Point {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "Point(x: {}, y: {}, z: {}, t: {})", self.x, self.y, self.z, self.t)
    }
}

/// a polynomial represented by its coefficients
#[derive(Debug, Clone)]
pub struct Polynomial {
    coeffs: Vec<f32>,
}

impl Polynomial {
    /// Returns an initialized polynomial
    /// 
    /// # Arguments
    /// 
    /// `coeffs` - vector of coefficients where index corresponds to power
    /// 
    /// # Examples
    /// 
    /// ```
    /// use trajgen::Polynomial;
    /// // f(t) = 7 - 2x + 3x^2
    /// let poly = Polynomial::new(&vec![7.0f32, -2.0f32, 3.0f32]);
    /// ```
    pub fn new(coeffs: &Vec<f32>) -> Polynomial {
        if coeffs.is_empty() {
            panic!("Cannot initialize Polynomial with empty coefficients!");
        }
        let coeffs = coeffs.clone();
        Polynomial { coeffs }
    }

    /// Returns the solution to f(t) at a given t
    /// 
    /// # Arguments
    /// 
    /// `t` - the value of t
    /// 
    /// # Examples
    /// 
    /// ```
    /// use trajgen::Polynomial;
    /// // f(t) = 7 - 2t + 3t^2
    /// let poly = Polynomial::new(&vec![7.0f32, -2.0f32, 3.0f32]);
    /// let ans = poly.eval(1.56);
    /// ```
    pub fn eval(&self, t: f32) -> f32 {
        let mut val = self.coeffs[0];
        let mut tt = t;
        for idx in 1..self.coeffs.len() {
            val = val + tt * self.coeffs[idx];
            tt *= t;
        }
        val
    }

    /// Returns the derivative of the polynomial
    /// 
    /// # Examples
    /// 
    /// ```
    /// use trajgen::Polynomial;
    /// // f(t) = 7 - 2t + 3t^2
    /// let poly = Polynomial::new(&vec![7.0f32, -2.0f32, 3.0f32]);
    /// // f'(t) = 2 + 6t
    /// let der = poly.derivative();
    /// ```
    pub fn derivative(&self) -> Polynomial {
        // special case when we are down to a single coefficient
        if self.coeffs.len() == 1 {
            return Polynomial::new(&vec![0.]);
        }

        let mut newcoeffs = Vec::new();

        // drop coefficient at index 0, and multiply coefficients by power
        for idx in 1..self.coeffs.len() {
            newcoeffs.push(self.coeffs[idx] * idx as f32);
        }

        Polynomial::new(&newcoeffs)
    }
}

impl fmt::Display for Polynomial {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let mut s = String::from("f(t) = ");
        for idx in 0..self.coeffs.len() {
            let coeff: String = match self.coeffs[idx].partial_cmp(&0.0)
                .expect("Found a coefficient that is NaN!") {
                Ordering::Less => format!("- {}", -self.coeffs[idx]),
                Ordering::Greater => format!("+ {}", self.coeffs[idx]),
                Ordering::Equal => String::new(),
            };

            // check if the coefficient is 0 - if so, just continue
            if coeff.len() == 0 {
                continue;
            }

            // append on the coefficient
            s.push_str(&coeff);

            // append on the x and it's power
            if idx > 0 {
                let x = if idx > 1 {format!("t^{} ", idx)} else {String::from("t ")};
                s.push_str(&x);
            } else {
                s.push(' ');
            }
        }
        write!(f, "{}", s.trim())
    }
}

/// The different types of supported trajectories
#[derive(Debug, PartialEq)]
pub enum TrajectoryType {
    Velocity,
    Acceleration,
    Jerk,
    Snap,
}

/// returns the number of coefficients required to solve for the given trajectory type
fn num_coeffs(traj_type: &TrajectoryType) -> usize {
    match traj_type {
        TrajectoryType::Velocity => 2,
        TrajectoryType::Acceleration => 4,
        TrajectoryType::Jerk => 6,
        TrajectoryType::Snap => 8,
    }
}

/// Trajectory Generator
#[derive(Debug)]
pub struct TrajectoryGenerator {
    traj_type: TrajectoryType,
    x_polys: Vec<Polynomial>,
    y_polys: Vec<Polynomial>,
    z_polys: Vec<Polynomial>,
    points: Vec<Point>,
}

impl TrajectoryGenerator {
    pub fn new(traj_type: TrajectoryType) -> TrajectoryGenerator {
        TrajectoryGenerator {
            traj_type,
            x_polys: Vec::new(),
            y_polys: Vec::new(),
            z_polys: Vec::new(),
            points: Vec::new(),
        }
    }

    /// Returns coefficients for position and time derivatives evaulated at a point in time
    /// Evaulates equations: c0*t^0 + c1*t^1 + ... + cn*t^n at some time t
    ///
    /// # Arguments
    ///
    /// `eqs` - position equation and time derivatives
    /// `t` - the time to evaulate at
    /// evaulates equations: c0*t^0 + c1*t^1 + ... + cn*t^n at some time t
    fn coeffs_at_time(&self, eqs: Vec<Polynomial>, t: f32) -> DMatrixf32 {
        let numcoeffs = num_coeffs(&self.traj_type);
        let mut retval = DMatrixf32::zeros(eqs.len(), numcoeffs);

        for ii in 0..eqs.len() {
            let mut row = DRowVectorf32::zeros(numcoeffs);
            let eq = &eqs[ii];
            // since the time derivatives will have less coefficients calculate an offset into the
            // resulting array to preserve the coefficients power
            let offset = numcoeffs - eq.coeffs.len();
            for jj in 0..eq.coeffs.len() {
                row[jj+offset] = eq.coeffs[jj] * t.powi(jj as i32);
            }

            retval.set_row(ii, &row);
        }

        retval
    }

    /// Generates the polynomials that will give a smooth path through the given points
    ///
    /// # Arguments
    ///
    /// `points` - list of trajectory waypoints ordered by time ascending
    ///
    /// # Examples
    ///
    /// ```
    /// use trajgen::{TrajectoryGenerator, TrajectoryType, Point};
    /// let mut traj_gen = TrajectoryGenerator::new(TrajectoryType::Jerk);
    /// // generates minimum jerk polynomials for a single segment trajectory
    /// let points: Vec<Point> = vec![Point::new(1., 2., 3., 0.), Point::new(2., 3., 4., 1.)];
    /// traj_gen.generate(&points);
    /// ```
    pub fn generate(&mut self, points: &Vec<Point>) {
        self.x_polys.clear();
        self.y_polys.clear();
        self.z_polys.clear();
        self.points.clear();

        self.points = points.clone();

        // make sure we have at least 2 points
        if points.len() < 2 {
            panic!("At least 2 points are required to generate a trajectory");
        }

        // check that times are incrementing
        for idx in 0..points.len()-1 {
            if points[idx+1].t <= points[idx].t {
                panic!("Time at index {} is less than or equal to time at index {}", idx+1, idx);
            }
        }

        info!("Generating trajectory using {} waypoints", points.len());

        let numcoeffs = num_coeffs(&self.traj_type);

        // use the polynomial class to create our coefficient equations and time derivatives
        // we want numcoeffs-1 equations total
        let mut eqs: Vec<Polynomial> = Vec::new();
        let poscoeffs:Vec<f32> = vec![1.0; numcoeffs];
        let poly = Polynomial::new(&poscoeffs);
        eqs.push(poly);
        while eqs.len() < numcoeffs - 1 {
            eqs.push(eqs.last().unwrap().derivative());
        }

        // create the A matrix
        let n= numcoeffs * (points.len() - 1);
        let mut a = DMatrixf32::zeros(n, n);

        // create b vectors for each of x, y, and z
        let mut bx = DColVectorf32::zeros(n);
        let mut by = DColVectorf32::zeros(n);
        let mut bz = DColVectorf32::zeros(n);

        let mut rowidx: usize = 0;
        let numeqs = (numcoeffs - 2) / 2;

        // fill in equations for first segment - time derivatives of position are all equal to 0 at start time
        let coeffs = self.coeffs_at_time(Vec::from(&eqs[1..1+numeqs]), points[0].t);
        let mut m = a.view_mut ((rowidx, 0), (numeqs, numcoeffs));
        for (idx, row) in coeffs.row_iter().enumerate() {
            let rowvec = DRowVectorf32::from(row);
            m.set_row(idx, &rowvec);
        }
        rowidx += numeqs;

        // fill in equations for last segment - time derivatives of position are all equal to 0 at end time
        let coeffs = self.coeffs_at_time(Vec::from(&eqs[1..1+numeqs]), points.last().unwrap().t);
        let mut m = a.view_mut ((rowidx, n - numcoeffs), (numeqs, numcoeffs));
        for (idx, row) in coeffs.row_iter().enumerate() {
            let rowvec = DRowVectorf32::from(row);
            m.set_row(idx, &rowvec);
        }
        rowidx += numeqs;

        // for each segment...
        for idx in 0..points.len()-1 {
            let startp = points[idx];
            let endp = points[idx+1];
            let startt = points[idx].t;
            let endt = points[idx+1].t;

            // fill in 2 equations for start and end point passing through the poly
            // start point
            let col = idx * numcoeffs;
            let coeffs = self.coeffs_at_time(Vec::from(&eqs[0..1]), startt);
            let rowvec = DRowVectorf32::from(coeffs.row(0));
            a.view_mut ((rowidx, col), (1, numcoeffs)).set_row(0, &rowvec);

            // set the b vector values
            bx[rowidx] = startp.x;
            by[rowidx] = startp.y;
            bz[rowidx] = startp.z;

            rowidx += 1;

            // end point
            let coeffs = self.coeffs_at_time(Vec::from(&eqs[0..1]), endt);
            let rowvec = DRowVectorf32::from(coeffs.row(0));
            a.view_mut ((rowidx, col), (1, numcoeffs)).set_row(0, &rowvec);

            // set the b vector values
            bx[rowidx] = endp.x;
            by[rowidx] = endp.y;
            bz[rowidx] = endp.z;

            rowidx += 1;
        }

        // for all segments, except last...
        for idx in 0..points.len()-2 {
            let endt = points[idx+1].t;
            let mut col = idx * numcoeffs;
            let mut coeffs = self.coeffs_at_time(Vec::from(&eqs[1..numcoeffs-1]), endt);
            let mut m = a.view_mut ((rowidx, col), (numcoeffs-2, numcoeffs));

            // fill in required equations for time derivatives to ensure they are the same through
            // the transition point
            for (ii, row) in coeffs.row_iter().enumerate() {
                let rowvec = DRowVectorf32::from(row);
                m.set_row(ii, &rowvec);
            }
            col += numcoeffs;

            // negate endt coefficients since we move everything to the lhs
            coeffs.neg_mut();
            let mut m = a.view_mut ((rowidx, col), (numcoeffs-2, numcoeffs));
            for (ii, row) in coeffs.row_iter().enumerate() {
                let rowvec = DRowVectorf32::from(row);
                m.set_row(ii, &rowvec);
            }
            rowidx += numeqs;
        }

        // invert the A matrix to solve the linear system Ax=b
        let ainv = a.try_inverse()
            .expect("Failed to invert A matrix!");
        let ansx = &ainv * bx;
        let ansy = &ainv * by;
        let ansz = &ainv * bz;

        // add a polynomial for each segment
        for idx in 0..points.len()-1 {
            let offset = idx * numcoeffs;

            // wasn't able to get this to work with slices...so this is a little ugly
            let mut arrx: Vec<f32> = Vec::new();
            let mut arry: Vec<f32> = Vec::new();
            let mut arrz: Vec<f32> = Vec::new();
            for ii in offset..offset+numcoeffs {
                arrx.push(ansx[(ii, 0)]);
                arry.push(ansy[(ii, 0)]);
                arrz.push(ansz[(ii, 0)]);
            }

            self.x_polys.push(Polynomial::new(&Vec::from(arrx)));
            self.y_polys.push(Polynomial::new(&Vec::from(arry)));
            self.z_polys.push(Polynomial::new(&Vec::from(arrz)));
        }

        info!("Finished generating trajectory using {} waypoints", points.len());
    }

    /// returns the index of the appropriate polygon to evaluate the value at time t
    fn poly_index(&self, t: f32) -> Option<usize> {
        if self.points.is_empty() {
            None
        } else {
            if t < self.points[0].t {
                Some(0)
            } else if t > self.points.last().unwrap().t {
                Some(self.x_polys.len() - 1)
            } else {
                let mut idx= 0;
                for ii in 0..self.points.len()-1 {
                    let startp = self.points[ii];
                    let endp = self.points[ii+1];
                    if t >= startp.t && t <= endp.t {
                        idx = ii;
                        break;
                    }
                }
                Some(idx)
            }
        }
    }

    /// Returns a `Point` structure with the value for the trajectory at time t
    ///
    /// # Arguments
    ///
    /// `t` - time
    /// `derivative` - derivative order, 0 for position, 1 for velocity, etc.
    ///
    /// # Examples
    ///
    /// ```
    /// use trajgen::{TrajectoryGenerator, TrajectoryType, Point};
    /// let mut traj_gen = TrajectoryGenerator::new(TrajectoryType::Jerk);
    /// let points: Vec<Point> = vec![Point::new(1., 2., 3., 0.), Point::new(2., 3., 4., 1.)];
    /// traj_gen.generate(&points);
    /// // get the position at t = 0.45
    /// let point = traj_gen.get_value(0.45, 0);
    /// ```
    pub fn get_value(&self, t: f32, derivative: u8) -> Option<Point> {
        let idx = self.poly_index(t)
            .expect("A trajectory must be generated first");

        let mut t = t;
        if t < self.points.first().unwrap().t {
            // if t is before start point time, clamp it to start time
            t = self.points.first().unwrap().t;
        } else if t > self.points.last().unwrap().t {
            // if t is after end point time, clamp it to end time
            t = self.points.last().unwrap().t;
        }

        let mut x_poly = self.x_polys[idx].clone();
        let mut y_poly = self.y_polys[idx].clone();
        let mut z_poly = self.z_polys[idx].clone();
        let mut derivative = derivative;
        loop {
            if derivative == 0 {
                break;
            }

            x_poly = x_poly.derivative();
            y_poly = y_poly.derivative();
            z_poly = z_poly.derivative();

            derivative -= 1;
        }

        // return the point evaluated at time t with the appropriate polynomial and derivative
        Some(Point{
            x: x_poly.eval(t),
            y: y_poly.eval(t),
            z: z_poly.eval(t),
            t
        })
    }

    /// Convenience function to return the position on the current trajectory
    pub fn get_position(&self, t: f32) -> Option<Point> {
        return self.get_value(t, 0);
    }

    /// Convenience function to return the velocity on the current trajectory
    pub fn get_velocity(&self, t: f32) -> Option<Point> {
        return self.get_value(t, 1);
    }

    /// Convenience function to return the accleration on the current trajectory
    pub fn get_acceleration(&self, t: f32) -> Option<Point> {
        return self.get_value(t, 2);
    }

    /// Convenience function to return the jerk on the current trajectory
    pub fn get_jerk(&self, t: f32) -> Option<Point> {
        return self.get_value(t, 3);
    }

    /// Returns an array of `Point` structures covering the input time range and step
    ///
    /// # Arguments
    ///
    /// `start` - start time
    /// `end` - end time
    /// `step` - time step
    /// `derivative` - derivative order, 0 for position, 1 for velocity, etc.
    ///
    /// # Examples
    ///
    /// ```
    /// use trajgen::{TrajectoryGenerator, TrajectoryType, Point};
    /// let mut traj_gen = TrajectoryGenerator::new(TrajectoryType::Jerk);
    /// let points: Vec<Point> = vec![Point::new(1., 2., 3., 0.), Point::new(2., 3., 4., 1.)];
    /// traj_gen.generate(&points);
    /// // get position values for time range [0, 1] using step of 0.1
    /// let positions = traj_gen.get_values(0., 1., 0.1, 0);
    /// ```
    pub fn get_values(&self, start: f32, end: f32, step: f32, derivative: u8) -> Vec<Point> {
        if end <= start {
            panic!("End must not be before start");
        }

        if step <= 0. {
            panic!("Step must be non-zero");
        }

        let mut values: Vec<Point> = Vec::new();
        let mut t = start;
        loop {
            values.push(self.get_value(t, derivative).unwrap());
            t += step;
            if t > end {
                break;
            }
        }

        values
    }

    /// Convenience function to sample positions on the current trajectory
    pub fn get_positions(&self, start: f32, end: f32, step: f32) -> Vec<Point> {
        return self.get_values(start, end, step, 0);
    }

    /// Convenience function to sample velocities on the current trajectory
    pub fn get_velocities(&self, start: f32, end: f32, step: f32) -> Vec<Point> {
        return self.get_values(start, end, step, 1);
    }

    /// Convenience function to sample accelerations on the current trajectory
    pub fn get_accelerations(&self, start: f32, end: f32, step: f32) -> Vec<Point> {
        return self.get_values(start, end, step,  2);
    }

    /// Convenience function to sample jerks on the current trajectory
    pub fn get_jerks(&self, start: f32, end: f32, step: f32) -> Vec<Point> {
        return self.get_values(start, end, step,  3);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use assert::close;

    #[test]
    fn point_new() {
        let point = Point::new(5.2, -6.75, 0.23, 2.32);
        assert_eq!(5.2, point.x);
        assert_eq!(-6.75, point.y);
        assert_eq!(0.23, point.z);
        assert_eq!(2.32, point.t);
    }

    #[test]
    fn point_fmt() {
        let point = Point::new(5.2, -6.75, 0.23, 2.32);
        assert_eq!("Point(x: 5.2, y: -6.75, z: 0.23, t: 2.32)", format!("{}", point))
    }

    #[test]
    fn coeffs_for_traj_type() {
        assert_eq!(2, num_coeffs(&TrajectoryType::Velocity));
        assert_eq!(4, num_coeffs(&TrajectoryType::Acceleration));
        assert_eq!(6, num_coeffs(&TrajectoryType::Jerk));
        assert_eq!(8, num_coeffs(&TrajectoryType::Snap));
    }

    #[test]
    fn poly_new() {
        // f(t) = 7 - 2t + 3t^2
        let poly = Polynomial::new(&vec![7.0f32, -2.0f32, 3.0f32]);

        // make sure it initialized correctly
        assert_eq!(3, poly.coeffs.len());
        assert_eq!(7.0, poly.coeffs[0]);
        assert_eq!(-2.0, poly.coeffs[1]);
        assert_eq!(3.0, poly.coeffs[2]);
    }

    #[test]
    #[should_panic]
    fn poly_new_invalid() {
        Polynomial::new(&Vec::new());
    }

    #[test]
    fn poly_eval() {
        // f(t) = 7 - 2t + 3t^2 - 4t^3
        let poly = Polynomial::new(&vec![7., -2., 3., -4.]);

        // f(0) = 7.0
        assert_eq!(7.0, poly.eval(0.0));

        // f(1) = 4.0
        assert_eq!(4.0, poly.eval(1.0));

        // f(7.26) = -1380.0261
        assert_eq!(-1380.0261, poly.eval(7.26));
    }

    #[test]
    fn poly_derivative() {
        // f(t) = 7 - 2t + 3t^2
        let poly = Polynomial::new(&vec![7.0f32, -2.0f32, 3.0f32]);
        let poly = poly.derivative();

        // f'(t) = -2 + 6t
        assert_eq!(2, poly.coeffs.len());
        assert_eq!(-2.0, poly.coeffs[0]);
        assert_eq!(6.0, poly.coeffs[1]);

        let poly = poly.derivative();

        // f''(t) = 6
        assert_eq!(1, poly.coeffs.len());
        assert_eq!(6.0, poly.coeffs[0]);

        let poly = poly.derivative();

        // f'''(t) = 0
        assert_eq!(1, poly.coeffs.len());
        assert_eq!(0.0, poly.coeffs[0]);
    }

    #[test]
    fn poly_print() {
        // f(t) = 7 - 2t + 3t^2
        let poly = Polynomial::new(&vec![7.0f32, -2.0f32, 3.0f32]);
        assert_eq!(poly.to_string(), "f(t) = + 7 - 2t + 3t^2");

        // f'(t) = -2 + 6t
        let poly = poly.derivative();
        assert_eq!(poly.to_string(), "f(t) = - 2 + 6t");

        // set a middle coefficient to 0
        // f(t) = 7 + 3t^2
        let poly = Polynomial::new(&vec![7.0f32, 0.0f32, 3.0f32]);
        assert_eq!(poly.to_string(), "f(t) = + 7 + 3t^2");

        // set the first coefficient to 0
        // f(t) = -2.1t + 3.9t^2
        let poly = Polynomial::new(&vec![0.0f32, -2.1f32, 3.9f32]);
        assert_eq!(poly.to_string(), "f(t) = - 2.1t + 3.9t^2");

        // set all coefficients to 0
        let poly = Polynomial::new(&vec![0.0f32, 0.0f32, 0.0f32]);
        assert_eq!(poly.to_string(), "f(t) =");
    }

    fn assert_poly_close(expected: Vec<f32>, actual: &Polynomial, delta: f32) {
        assert_eq!(expected.len(), actual.coeffs.len());
        for idx in 0..expected.len() {
            close(expected[idx], actual.coeffs[idx], delta);
        }
    }

    #[test]
    fn traj_gen_new() {
        let traj_gen = TrajectoryGenerator::new(TrajectoryType::Jerk);
        assert_eq!(TrajectoryType::Jerk, traj_gen.traj_type);
        assert_eq!(0, traj_gen.x_polys.len());
        assert_eq!(0, traj_gen.y_polys.len());
        assert_eq!(0, traj_gen.z_polys.len());
        assert_eq!(0, traj_gen.points.len());
    }

    #[test]
    #[should_panic]
    fn traj_generate_empty_inputs() {
        let mut traj_gen = TrajectoryGenerator::new(TrajectoryType::Jerk);
        let points: Vec<Point> = Vec::new();
        traj_gen.generate(&points);
    }

    #[test]
    #[should_panic]
    fn traj_generate_single_point() {
        let mut traj_gen = TrajectoryGenerator::new(TrajectoryType::Jerk);
        // only 1 point and 2 times
        let points: Vec<Point> = vec![Point::new(1., 2., 3., 0.)];
        traj_gen.generate(&points);
    }

    #[test]
    fn traj_generate_two_points_velocity() {
        let mut traj_gen = TrajectoryGenerator::new(TrajectoryType::Velocity);
        let points: Vec<Point> = vec![Point::new(1., 2., 3., 0.),
                                      Point::new(2., 3., 4., 1.)];
        traj_gen.generate(&points);

        // 2 points will only generate a single segment
        assert_eq!(1, traj_gen.x_polys.len());
        assert_eq!(1, traj_gen.y_polys.len());
        assert_eq!(1, traj_gen.z_polys.len());

        // check that each poly is close to what we expect
        let delta = 1e-9;
        assert_poly_close(vec![1., 1.], &traj_gen.x_polys[0], delta);
        assert_poly_close(vec![2., 1.], &traj_gen.y_polys[0], delta);
        assert_poly_close(vec![3., 1.], &traj_gen.z_polys[0], delta);

        // ensure that the trajectory passes through the input points
        close(1., traj_gen.x_polys[0].eval(0.), delta);
        close(2., traj_gen.y_polys[0].eval(0.), delta);
        close(3., traj_gen.z_polys[0].eval(0.), delta);
        close(2., traj_gen.x_polys[0].eval(1.), delta);
        close(3., traj_gen.y_polys[0].eval(1.), delta);
        close(4., traj_gen.z_polys[0].eval(1.), delta);
    }

    #[test]
    fn traj_generate_two_points_acceleration() {
        let mut traj_gen = TrajectoryGenerator::new(TrajectoryType::Acceleration);
        let points: Vec<Point> = vec![Point::new(1., 2., 3., 0.), Point::new(2., 3., 4., 1.)];
        traj_gen.generate(&points);

        // 2 points will only generate a single segment
        assert_eq!(1, traj_gen.x_polys.len());
        assert_eq!(1, traj_gen.y_polys.len());
        assert_eq!(1, traj_gen.z_polys.len());

        // check that each poly is close to what we expect
        let delta = 1e-9;
        assert_poly_close(vec![1., 0., 3., -2.], &traj_gen.x_polys[0], delta);
        assert_poly_close(vec![2., 0., 3., -2.], &traj_gen.y_polys[0], delta);
        assert_poly_close(vec![3., 0., 3., -2.], &traj_gen.z_polys[0], delta);

        // ensure that the trajectory passes through the input points
        close(1., traj_gen.x_polys[0].eval(0.), delta);
        close(2., traj_gen.y_polys[0].eval(0.), delta);
        close(3., traj_gen.z_polys[0].eval(0.), delta);
        close(2., traj_gen.x_polys[0].eval(1.), delta);
        close(3., traj_gen.y_polys[0].eval(1.), delta);
        close(4., traj_gen.z_polys[0].eval(1.), delta);
    }

    #[test]
    fn traj_generate_two_points_jerk() {
        let mut traj_gen = TrajectoryGenerator::new(TrajectoryType::Jerk);
        let points: Vec<Point> = vec![Point::new(1., 2., 3., 0.), Point::new(2., 3., 4., 1.)];
        traj_gen.generate(&points);

        // 2 points will only generate a single segment
        assert_eq!(1, traj_gen.x_polys.len());
        assert_eq!(1, traj_gen.y_polys.len());
        assert_eq!(1, traj_gen.z_polys.len());

        // check that each poly is close to what we expect
        let delta = 1e-4;
        assert_poly_close(vec![1., 0., 0., 10., -15., 6.], &traj_gen.x_polys[0], delta);
        assert_poly_close(vec![2., 0., 0., 10., -15., 6.], &traj_gen.y_polys[0], delta);
        assert_poly_close(vec![3., 0., 0., 10., -15., 6.], &traj_gen.z_polys[0], delta);

        // ensure that the trajectory passes through the input points
        close(1., traj_gen.x_polys[0].eval(0.), delta);
        close(2., traj_gen.y_polys[0].eval(0.), delta);
        close(3., traj_gen.z_polys[0].eval(0.), delta);
        close(2., traj_gen.x_polys[0].eval(1.), delta);
        close(3., traj_gen.y_polys[0].eval(1.), delta);
        close(4., traj_gen.z_polys[0].eval(1.), delta);
    }

    #[test]
    fn traj_generate_two_points_snap() {
        let mut traj_gen = TrajectoryGenerator::new(TrajectoryType::Snap);
        let points: Vec<Point> = vec![Point::new(1., 2., 3., 0.), Point::new(2., 3., 4., 1.)];
        traj_gen.generate(&points);

        // 2 points will only generate a single segment
        assert_eq!(1, traj_gen.x_polys.len());
        assert_eq!(1, traj_gen.y_polys.len());
        assert_eq!(1, traj_gen.z_polys.len());

        // check that each poly is close to what we expect
        let delta = 1e-2;
        assert_poly_close(vec![1., 0., 0., 0., 35., -84., 70., -20.], &traj_gen.x_polys[0], delta);
        assert_poly_close(vec![2., 0., 0., 0., 35., -84., 70., -20.], &traj_gen.y_polys[0], delta);
        assert_poly_close(vec![3., 0., 0., 0., 35., -84., 70., -20.], &traj_gen.z_polys[0], delta);

        // ensure that the trajectory passes through the input points
        close(1., traj_gen.x_polys[0].eval(0.), delta);
        close(2., traj_gen.y_polys[0].eval(0.), delta);
        close(3., traj_gen.z_polys[0].eval(0.), delta);
        close(2., traj_gen.x_polys[0].eval(1.), delta);
        close(3., traj_gen.y_polys[0].eval(1.), delta);
        close(4., traj_gen.z_polys[0].eval(1.), delta);
    }

    #[test]
    fn traj_generate_multi_points_velocity() {
        let mut traj_gen = TrajectoryGenerator::new(TrajectoryType::Velocity);
        let points: Vec<Point> = vec![Point::new(1., 2., 3., 0.),
                                      Point::new(2., 3., 4., 1.),
                                      Point::new(3., 4., 5., 2.)];
        traj_gen.generate(&points);

        // should be 2 segments since we have 3 points
        assert_eq!(2, traj_gen.x_polys.len());
        assert_eq!(2, traj_gen.y_polys.len());
        assert_eq!(2, traj_gen.z_polys.len());

        // check that each poly is close to what we expect
        let delta = 1e-9;
        assert_poly_close(vec![1., 1.], &traj_gen.x_polys[0], delta);
        assert_poly_close(vec![1., 1.], &traj_gen.x_polys[1], delta);
        assert_poly_close(vec![2., 1.], &traj_gen.y_polys[0], delta);
        assert_poly_close(vec![2., 1.], &traj_gen.y_polys[1], delta);
        assert_poly_close(vec![3., 1.], &traj_gen.z_polys[0], delta);
        assert_poly_close(vec![3., 1.], &traj_gen.z_polys[1], delta);

        // ensure that the trajectory passes through the input points and that the intermediary
        // point works for both polynomials
        close(1., traj_gen.x_polys[0].eval(0.), delta);
        close(2., traj_gen.y_polys[0].eval(0.), delta);
        close(3., traj_gen.z_polys[0].eval(0.), delta);
        close(2., traj_gen.x_polys[0].eval(1.), delta);
        close(3., traj_gen.y_polys[0].eval(1.), delta);
        close(4., traj_gen.z_polys[0].eval(1.), delta);
        close(2., traj_gen.x_polys[1].eval(1.), delta);
        close(3., traj_gen.y_polys[1].eval(1.), delta);
        close(4., traj_gen.z_polys[1].eval(1.), delta);
        close(3., traj_gen.x_polys[1].eval(2.), delta);
        close(4., traj_gen.y_polys[1].eval(2.), delta);
        close(5., traj_gen.z_polys[1].eval(2.), delta);
    }

    #[test]
    fn traj_generate_multi_points_acceleration() {
        let mut traj_gen = TrajectoryGenerator::new(TrajectoryType::Acceleration);
        let points: Vec<Point> = vec![Point::new(1., 2., 3., 0.),
                                      Point::new(2., 3., 4., 1.),
                                      Point::new(3., 4., 5., 2.)];
        traj_gen.generate(&points);

        // should be 2 segments since we have 3 points
        assert_eq!(2, traj_gen.x_polys.len());
        assert_eq!(2, traj_gen.y_polys.len());
        assert_eq!(2, traj_gen.z_polys.len());

        // check that each poly is close to what we expect
        let delta = 1e-4;
        assert_poly_close(vec![1., 0., 1.5, -0.5], &traj_gen.x_polys[0], delta);
        assert_poly_close(vec![1., 0., 1.5, -0.5], &traj_gen.x_polys[1], delta);
        assert_poly_close(vec![2., 0., 1.5, -0.5], &traj_gen.y_polys[0], delta);
        assert_poly_close(vec![2., 0., 1.5, -0.5], &traj_gen.y_polys[1], delta);
        assert_poly_close(vec![3., 0., 1.5, -0.5], &traj_gen.z_polys[0], delta);
        assert_poly_close(vec![3., 0., 1.5, -0.5], &traj_gen.z_polys[1], delta);

        // ensure that the trajectory passes through the input points and that the intermediary
        // point works for both polynomials
        close(1., traj_gen.x_polys[0].eval(0.), delta);
        close(2., traj_gen.y_polys[0].eval(0.), delta);
        close(3., traj_gen.z_polys[0].eval(0.), delta);
        close(2., traj_gen.x_polys[0].eval(1.), delta);
        close(3., traj_gen.y_polys[0].eval(1.), delta);
        close(4., traj_gen.z_polys[0].eval(1.), delta);
        close(2., traj_gen.x_polys[1].eval(1.), delta);
        close(3., traj_gen.y_polys[1].eval(1.), delta);
        close(4., traj_gen.z_polys[1].eval(1.), delta);
        close(3., traj_gen.x_polys[1].eval(2.), delta);
        close(4., traj_gen.y_polys[1].eval(2.), delta);
        close(5., traj_gen.z_polys[1].eval(2.), delta);
    }

    #[test]
    fn traj_generate_multi_points_jerk() {
        let mut traj_gen = TrajectoryGenerator::new(TrajectoryType::Jerk);
        let points: Vec<Point> = vec![Point::new(1., 2., 3., 0.),
                                      Point::new(2., 3., 4., 1.),
                                      Point::new(3., 4., 5., 2.)];
        traj_gen.generate(&points);

        // should be 2 segments since we have 3 points
        assert_eq!(2, traj_gen.x_polys.len());
        assert_eq!(2, traj_gen.y_polys.len());
        assert_eq!(2, traj_gen.z_polys.len());

        // check that each poly is close to what we expect
        let delta = 1e-4;
        assert_poly_close(vec![1., 0., 0., 2.5, -1.875, 0.375], &traj_gen.x_polys[0], delta);
        assert_poly_close(vec![1., 0., 0., 2.5, -1.875, 0.375], &traj_gen.x_polys[1], delta);
        assert_poly_close(vec![2., 0., 0., 2.5, -1.875, 0.375], &traj_gen.y_polys[0], delta);
        assert_poly_close(vec![2., 0., 0., 2.5, -1.875, 0.375], &traj_gen.y_polys[1], delta);
        assert_poly_close(vec![3., 0., 0., 2.5, -1.875, 0.375], &traj_gen.z_polys[0], delta);
        assert_poly_close(vec![3., 0., 0., 2.5, -1.875, 0.375], &traj_gen.z_polys[1], delta);

        // ensure that the trajectory passes through the input points and that the intermediary
        // point works for both polynomials
        close(1., traj_gen.x_polys[0].eval(0.), delta);
        close(2., traj_gen.y_polys[0].eval(0.), delta);
        close(3., traj_gen.z_polys[0].eval(0.), delta);
        close(2., traj_gen.x_polys[0].eval(1.), delta);
        close(3., traj_gen.y_polys[0].eval(1.), delta);
        close(4., traj_gen.z_polys[0].eval(1.), delta);
        close(2., traj_gen.x_polys[1].eval(1.), delta);
        close(3., traj_gen.y_polys[1].eval(1.), delta);
        close(4., traj_gen.z_polys[1].eval(1.), delta);
        close(3., traj_gen.x_polys[1].eval(2.), delta);
        close(4., traj_gen.y_polys[1].eval(2.), delta);
        close(5., traj_gen.z_polys[1].eval(2.), delta);
    }

    #[test]
    fn traj_generate_multi_points_snap() {
        let mut traj_gen = TrajectoryGenerator::new(TrajectoryType::Snap);
        let points: Vec<Point> = vec![Point::new(1., 2., 3., 0.),
                                      Point::new(2., 3., 4., 1.),
                                      Point::new(3., 4., 5., 2.)];
        traj_gen.generate(&points);

        // should be 2 segments since we have 3 points
        assert_eq!(2, traj_gen.x_polys.len());
        assert_eq!(2, traj_gen.y_polys.len());
        assert_eq!(2, traj_gen.z_polys.len());

        // check that each poly is close to what we expect
        let delta = 1e-2;
        assert_poly_close(vec![1., 0., 0., 0., 4.375, -5.25, 2.1875, -0.3125], &traj_gen.x_polys[0], delta);
        assert_poly_close(vec![1., 0., 0., 0., 4.375, -5.25, 2.1875, -0.3125], &traj_gen.x_polys[1], delta);
        assert_poly_close(vec![2., 0., 0., 0., 4.375, -5.25, 2.1875, -0.3125], &traj_gen.y_polys[0], delta);
        assert_poly_close(vec![2., 0., 0., 0., 4.375, -5.25, 2.1875, -0.3125], &traj_gen.y_polys[1], delta);
        assert_poly_close(vec![3., 0., 0., 0., 4.375, -5.25, 2.1875, -0.3125], &traj_gen.z_polys[0], delta);
        assert_poly_close(vec![3., 0., 0., 0., 4.375, -5.25, 2.1875, -0.3125], &traj_gen.z_polys[1], delta);

        // ensure that the trajectory passes through the input points and that the intermediary
        // point works for both polynomials
        close(1., traj_gen.x_polys[0].eval(0.), delta);
        close(2., traj_gen.y_polys[0].eval(0.), delta);
        close(3., traj_gen.z_polys[0].eval(0.), delta);
        close(2., traj_gen.x_polys[0].eval(1.), delta);
        close(3., traj_gen.y_polys[0].eval(1.), delta);
        close(4., traj_gen.z_polys[0].eval(1.), delta);
        close(2., traj_gen.x_polys[1].eval(1.), delta);
        close(3., traj_gen.y_polys[1].eval(1.), delta);
        close(4., traj_gen.z_polys[1].eval(1.), delta);
        close(3., traj_gen.x_polys[1].eval(2.), delta);
        close(4., traj_gen.y_polys[1].eval(2.), delta);
        close(5., traj_gen.z_polys[1].eval(2.), delta);
    }

    #[test]
    #[should_panic]
    fn traj_poly_index_invalid() {
        let traj_gen = TrajectoryGenerator::new(TrajectoryType::Snap);
        traj_gen.poly_index(1.23).expect("This should fail");
    }

    #[test]
    fn traj_poly_index() {
        let mut traj_gen = TrajectoryGenerator::new(TrajectoryType::Snap);
        let points: Vec<Point> = vec![Point::new(1., 2., 3., 0.),
                                      Point::new(2., 3., 4., 1.),
                                      Point::new(3., 4., 5., 2.)];
        traj_gen.generate(&points);

        // before start time
        assert_eq!(0, traj_gen.poly_index(-1.23).unwrap());

        // after start time
        assert_eq!(1, traj_gen.poly_index(5.23).unwrap());

        // between points 1-2
        assert_eq!(0, traj_gen.poly_index(0.5).unwrap());

        // between points 2-3
        assert_eq!(1, traj_gen.poly_index(1.5).unwrap());

        // on the start point time
        assert_eq!(0, traj_gen.poly_index(0.).unwrap());

        // on the middle point time
        assert_eq!(0, traj_gen.poly_index(1.).unwrap());

        // on the end point time
        assert_eq!(1, traj_gen.poly_index(2.).unwrap());
    }

    #[test]
    fn traj_get_value() {
        let mut traj_gen = TrajectoryGenerator::new(TrajectoryType::Snap);
        let points: Vec<Point> = vec![Point::new(1., 2., 3., 0.),
                                      Point::new(2., 3., 4., 1.),
                                      Point::new(3., 4., 5., 2.)];
        traj_gen.generate(&points);

        let delta = 1e-3;

        // position
        close(1., traj_gen.get_value(0., 0).unwrap().x, delta);
        close(2., traj_gen.get_value(0., 0).unwrap().y, delta);
        close(3., traj_gen.get_value(0., 0).unwrap().z, delta);
        close(2., traj_gen.get_value(1., 0).unwrap().x, delta);
        close(3., traj_gen.get_value(1., 0).unwrap().y, delta);
        close(4., traj_gen.get_value(1., 0).unwrap().z, delta);
        close(3., traj_gen.get_value(2., 0).unwrap().x, delta);
        close(4., traj_gen.get_value(2., 0).unwrap().y, delta);
        close(5., traj_gen.get_value(2., 0).unwrap().z, delta);

        // velocity
        close(2.187, traj_gen.get_value(1., 1).unwrap().x, delta);
        close(2.187, traj_gen.get_value(1., 1).unwrap().y, delta);
        close(2.187, traj_gen.get_value(1., 1).unwrap().z, delta);

        // acceleration should be zero at the midpoint for this trajectory
        close(0., traj_gen.get_value(1., 2).unwrap().x, delta);
        close(0., traj_gen.get_value(1., 2).unwrap().y, delta);
        close(0., traj_gen.get_value(1., 2).unwrap().z, delta);
    }

    #[test]
    fn traj_convenience() {
        let mut traj_gen = TrajectoryGenerator::new(TrajectoryType::Snap);
        let points: Vec<Point> = vec![Point::new(1., 2., 3., 0.),
                                      Point::new(2., 3., 4., 1.),
                                      Point::new(3., 4., 5., 2.)];
        traj_gen.generate(&points);

        let delta = 1e-2;
        close(3., traj_gen.get_position(1.).unwrap().y, delta);

        // higher derivatives should have zero value at start and end
        close(0., traj_gen.get_velocity(0.).unwrap().y, delta);
        close(0., traj_gen.get_acceleration(0.).unwrap().z, delta);
        close(0., traj_gen.get_jerk(0.).unwrap().x, delta);
        close(0., traj_gen.get_velocity(2.).unwrap().y, delta);
        close(0., traj_gen.get_acceleration(2.).unwrap().z, delta);
        close(0., traj_gen.get_jerk(2.).unwrap().x, delta);
    }

    #[test]
    fn traj_get_values() {
        let mut traj_gen = TrajectoryGenerator::new(TrajectoryType::Snap);
        let points: Vec<Point> = vec![Point::new(1., 2., 3., 0.),
                                      Point::new(2., 3., 4., 1.),
                                      Point::new(3., 4., 5., 2.)];
        traj_gen.generate(&points);

        let points = traj_gen.get_values(0., 2., 0.2, 0);
        assert_eq!(10, points.len());

        // should also be able to sample outside of the range with no issues
        let points = traj_gen.get_values(-1., 3., 0.2, 0);
        assert_eq!(20, points.len());
    }

    #[test]
    #[should_panic]
    fn traj_get_values_bad_range() {
        let mut traj_gen = TrajectoryGenerator::new(TrajectoryType::Snap);
        let points: Vec<Point> = vec![Point::new(1., 2., 3., 0.),
                                      Point::new(2., 3., 4., 1.),
                                      Point::new(3., 4., 5., 2.)];
        traj_gen.generate(&points);

        traj_gen.get_values(2., 1., 0.2, 0);
    }

    #[test]
    #[should_panic]
    fn traj_get_values_bad_step() {
        let mut traj_gen = TrajectoryGenerator::new(TrajectoryType::Snap);
        let points: Vec<Point> = vec![Point::new(1., 2., 3., 0.),
                                      Point::new(2., 3., 4., 1.),
                                      Point::new(3., 4., 5., 2.)];
        traj_gen.generate(&points);

        traj_gen.get_values(2., 1., -0.2, 0);
    }

    #[test]
    fn traj_convenience_sample() {
        let mut traj_gen = TrajectoryGenerator::new(TrajectoryType::Snap);
        let points: Vec<Point> = vec![Point::new(1., 2., 3., 0.),
                                      Point::new(2., 3., 4., 1.),
                                      Point::new(3., 4., 5., 2.)];
        traj_gen.generate(&points);

        assert_eq!(10, traj_gen.get_positions(0., 2., 0.2).len());
        assert_eq!(10, traj_gen.get_velocities(0., 2., 0.2).len());
        assert_eq!(10, traj_gen.get_accelerations(0., 2., 0.2).len());
        assert_eq!(10, traj_gen.get_jerks(0., 2., 0.2).len());
    }
}
