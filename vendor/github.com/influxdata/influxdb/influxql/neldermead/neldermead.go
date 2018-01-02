// This is an implementation of the Nelder-Mead optimization method
// Based on work by Michael F. Hutt http://www.mikehutt.com/neldermead.html
package neldermead

import "math"

const (
	defaultMaxIterations = 1000
	// reflection coefficient
	defaultAlpha = 1.0
	// contraction coefficient
	defaultBeta = 0.5
	// expansion coefficient
	defaultGamma = 2.0
)

type Optimizer struct {
	MaxIterations int
	// reflection coefficient
	Alpha,
	// contraction coefficient
	Beta,
	// expansion coefficient
	Gamma float64
}

func New() *Optimizer {
	return &Optimizer{
		MaxIterations: defaultMaxIterations,
		Alpha:         defaultAlpha,
		Beta:          defaultBeta,
		Gamma:         defaultGamma,
	}
}

func (o *Optimizer) Optimize(
	objfunc func([]float64) float64,
	start []float64,
	epsilon,
	scale float64,
) (float64, []float64) {
	n := len(start)

	//holds vertices of simplex
	v := make([][]float64, n+1)
	for i := range v {
		v[i] = make([]float64, n)
	}

	//value of function at each vertex
	f := make([]float64, n+1)

	//reflection - coordinates
	vr := make([]float64, n)

	//expansion - coordinates
	ve := make([]float64, n)

	//contraction - coordinates
	vc := make([]float64, n)

	//centroid - coordinates
	vm := make([]float64, n)

	// create the initial simplex
	// assume one of the vertices is 0,0

	pn := scale * (math.Sqrt(float64(n+1)) - 1 + float64(n)) / (float64(n) * math.Sqrt(2))
	qn := scale * (math.Sqrt(float64(n+1)) - 1) / (float64(n) * math.Sqrt(2))

	for i := 0; i < n; i++ {
		v[0][i] = start[i]
	}

	for i := 1; i <= n; i++ {
		for j := 0; j < n; j++ {
			if i-1 == j {
				v[i][j] = pn + start[j]
			} else {
				v[i][j] = qn + start[j]
			}
		}
	}

	// find the initial function values
	for j := 0; j <= n; j++ {
		f[j] = objfunc(v[j])
	}

	// begin the main loop of the minimization
	for itr := 1; itr <= o.MaxIterations; itr++ {

		// find the indexes of the largest and smallest values
		vg := 0
		vs := 0
		for i := 0; i <= n; i++ {
			if f[i] > f[vg] {
				vg = i
			}
			if f[i] < f[vs] {
				vs = i
			}
		}
		// find the index of the second largest value
		vh := vs
		for i := 0; i <= n; i++ {
			if f[i] > f[vh] && f[i] < f[vg] {
				vh = i
			}
		}

		// calculate the centroid
		for i := 0; i <= n-1; i++ {
			cent := 0.0
			for m := 0; m <= n; m++ {
				if m != vg {
					cent += v[m][i]
				}
			}
			vm[i] = cent / float64(n)
		}

		// reflect vg to new vertex vr
		for i := 0; i <= n-1; i++ {
			vr[i] = vm[i] + o.Alpha*(vm[i]-v[vg][i])
		}

		// value of function at reflection point
		fr := objfunc(vr)

		if fr < f[vh] && fr >= f[vs] {
			for i := 0; i <= n-1; i++ {
				v[vg][i] = vr[i]
			}
			f[vg] = fr
		}

		// investigate a step further in this direction
		if fr < f[vs] {
			for i := 0; i <= n-1; i++ {
				ve[i] = vm[i] + o.Gamma*(vr[i]-vm[i])
			}

			// value of function at expansion point
			fe := objfunc(ve)

			// by making fe < fr as opposed to fe < f[vs],
			// Rosenbrocks function takes 63 iterations as opposed
			// to 64 when using double variables.

			if fe < fr {
				for i := 0; i <= n-1; i++ {
					v[vg][i] = ve[i]
				}
				f[vg] = fe
			} else {
				for i := 0; i <= n-1; i++ {
					v[vg][i] = vr[i]
				}
				f[vg] = fr
			}
		}

		// check to see if a contraction is necessary
		if fr >= f[vh] {
			if fr < f[vg] && fr >= f[vh] {
				// perform outside contraction
				for i := 0; i <= n-1; i++ {
					vc[i] = vm[i] + o.Beta*(vr[i]-vm[i])
				}
			} else {
				// perform inside contraction
				for i := 0; i <= n-1; i++ {
					vc[i] = vm[i] - o.Beta*(vm[i]-v[vg][i])
				}
			}

			// value of function at contraction point
			fc := objfunc(vc)

			if fc < f[vg] {
				for i := 0; i <= n-1; i++ {
					v[vg][i] = vc[i]
				}
				f[vg] = fc
			} else {
				// at this point the contraction is not successful,
				// we must halve the distance from vs to all the
				// vertices of the simplex and then continue.

				for row := 0; row <= n; row++ {
					if row != vs {
						for i := 0; i <= n-1; i++ {
							v[row][i] = v[vs][i] + (v[row][i]-v[vs][i])/2.0
						}
					}
				}
				f[vg] = objfunc(v[vg])
				f[vh] = objfunc(v[vh])
			}
		}

		// test for convergence
		fsum := 0.0
		for i := 0; i <= n; i++ {
			fsum += f[i]
		}
		favg := fsum / float64(n+1)
		s := 0.0
		for i := 0; i <= n; i++ {
			s += math.Pow((f[i]-favg), 2.0) / float64(n)
		}
		s = math.Sqrt(s)
		if s < epsilon {
			break
		}
	}

	// find the index of the smallest value
	vs := 0
	for i := 0; i <= n; i++ {
		if f[i] < f[vs] {
			vs = i
		}
	}

	parameters := make([]float64, n)
	for i := 0; i < n; i++ {
		parameters[i] = v[vs][i]
	}

	min := objfunc(v[vs])

	return min, parameters
}
