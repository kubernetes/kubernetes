// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package gldriver provides an OpenGL driver for accessing a screen.
package gldriver // import "golang.org/x/exp/shiny/driver/gldriver"

import (
	"encoding/binary"
	"fmt"
	"math"

	"golang.org/x/exp/shiny/driver/internal/errscreen"
	"golang.org/x/exp/shiny/screen"
	"golang.org/x/image/math/f64"
	"golang.org/x/mobile/gl"
)

// Main is called by the program's main function to run the graphical
// application.
//
// It calls f on the Screen, possibly in a separate goroutine, as some OS-
// specific libraries require being on 'the main thread'. It returns when f
// returns.
func Main(f func(screen.Screen)) {
	if err := main(f); err != nil {
		f(errscreen.Stub(err))
	}
}

func mul(a, b f64.Aff3) f64.Aff3 {
	return f64.Aff3{
		a[0]*b[0] + a[1]*b[3],
		a[0]*b[1] + a[1]*b[4],
		a[0]*b[2] + a[1]*b[5] + a[2],

		a[3]*b[0] + a[4]*b[3],
		a[3]*b[1] + a[4]*b[4],
		a[3]*b[2] + a[4]*b[5] + a[5],
	}
}

// writeAff3 must only be called while holding windowImpl.glctxMu.
func writeAff3(glctx gl.Context, u gl.Uniform, a f64.Aff3) {
	var m [9]float32
	m[0*3+0] = float32(a[0*3+0])
	m[0*3+1] = float32(a[1*3+0])
	m[0*3+2] = 0
	m[1*3+0] = float32(a[0*3+1])
	m[1*3+1] = float32(a[1*3+1])
	m[1*3+2] = 0
	m[2*3+0] = float32(a[0*3+2])
	m[2*3+1] = float32(a[1*3+2])
	m[2*3+2] = 1
	glctx.UniformMatrix3fv(u, m[:])
}

// f32Bytes returns the byte representation of float32 values in the given byte
// order. byteOrder must be either binary.BigEndian or binary.LittleEndian.
func f32Bytes(byteOrder binary.ByteOrder, values ...float32) []byte {
	le := false
	switch byteOrder {
	case binary.BigEndian:
	case binary.LittleEndian:
		le = true
	default:
		panic(fmt.Sprintf("invalid byte order %v", byteOrder))
	}

	b := make([]byte, 4*len(values))
	for i, v := range values {
		u := math.Float32bits(v)
		if le {
			b[4*i+0] = byte(u >> 0)
			b[4*i+1] = byte(u >> 8)
			b[4*i+2] = byte(u >> 16)
			b[4*i+3] = byte(u >> 24)
		} else {
			b[4*i+0] = byte(u >> 24)
			b[4*i+1] = byte(u >> 16)
			b[4*i+2] = byte(u >> 8)
			b[4*i+3] = byte(u >> 0)
		}
	}
	return b
}

// compileProgram must only be called while holding windowImpl.glctxMu.
func compileProgram(glctx gl.Context, vSrc, fSrc string) (gl.Program, error) {
	program := glctx.CreateProgram()
	if program.Value == 0 {
		return gl.Program{}, fmt.Errorf("gldriver: no programs available")
	}

	vertexShader, err := compileShader(glctx, gl.VERTEX_SHADER, vSrc)
	if err != nil {
		return gl.Program{}, err
	}
	fragmentShader, err := compileShader(glctx, gl.FRAGMENT_SHADER, fSrc)
	if err != nil {
		glctx.DeleteShader(vertexShader)
		return gl.Program{}, err
	}

	glctx.AttachShader(program, vertexShader)
	glctx.AttachShader(program, fragmentShader)
	glctx.LinkProgram(program)

	// Flag shaders for deletion when program is unlinked.
	glctx.DeleteShader(vertexShader)
	glctx.DeleteShader(fragmentShader)

	if glctx.GetProgrami(program, gl.LINK_STATUS) == 0 {
		defer glctx.DeleteProgram(program)
		return gl.Program{}, fmt.Errorf("gldriver: program compile: %s", glctx.GetProgramInfoLog(program))
	}
	return program, nil
}

// compileShader must only be called while holding windowImpl.glctxMu.
func compileShader(glctx gl.Context, shaderType gl.Enum, src string) (gl.Shader, error) {
	shader := glctx.CreateShader(shaderType)
	if shader.Value == 0 {
		return gl.Shader{}, fmt.Errorf("gldriver: could not create shader (type %v)", shaderType)
	}
	glctx.ShaderSource(shader, src)
	glctx.CompileShader(shader)
	if glctx.GetShaderi(shader, gl.COMPILE_STATUS) == 0 {
		defer glctx.DeleteShader(shader)
		return gl.Shader{}, fmt.Errorf("gldriver: shader compile: %s", glctx.GetShaderInfoLog(shader))
	}
	return shader, nil
}
