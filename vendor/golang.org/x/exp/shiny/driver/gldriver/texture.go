// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gldriver

import (
	"encoding/binary"
	"image"
	"image/color"
	"image/draw"

	"golang.org/x/exp/shiny/screen"
	"golang.org/x/mobile/gl"
)

type textureImpl struct {
	w    *windowImpl
	id   gl.Texture
	size image.Point
}

func (t *textureImpl) Size() image.Point       { return t.size }
func (t *textureImpl) Bounds() image.Rectangle { return image.Rectangle{Max: t.size} }

func (t *textureImpl) Release() {
	t.w.glctxMu.Lock()
	defer t.w.glctxMu.Unlock()

	t.w.glctx.DeleteTexture(t.id)
	t.id = gl.Texture{}
}

func (t *textureImpl) Upload(dp image.Point, src screen.Buffer, sr image.Rectangle) {
	buf := src.(*bufferImpl)
	buf.preUpload()

	// src2dst is added to convert from the src coordinate space to the dst
	// coordinate space. It is subtracted to convert the other way.
	src2dst := dp.Sub(sr.Min)

	// Clip to the source.
	sr = sr.Intersect(buf.Bounds())

	// Clip to the destination.
	dr := sr.Add(src2dst)
	dr = dr.Intersect(t.Bounds())
	if dr.Empty() {
		return
	}

	// Bring dr.Min in dst-space back to src-space to get the pixel buffer offset.
	pix := buf.rgba.Pix[buf.rgba.PixOffset(dr.Min.X-src2dst.X, dr.Min.Y-src2dst.Y):]

	t.w.glctxMu.Lock()
	defer t.w.glctxMu.Unlock()

	t.w.glctx.BindTexture(gl.TEXTURE_2D, t.id)

	width := dr.Dx()
	if width*4 == buf.rgba.Stride {
		t.w.glctx.TexSubImage2D(gl.TEXTURE_2D, 0, dr.Min.X, dr.Min.Y, width, dr.Dy(), gl.RGBA, gl.UNSIGNED_BYTE, pix)
		return
	}
	// TODO: can we use GL_UNPACK_ROW_LENGTH with glPixelStorei for stride in
	// ES 3.0, instead of uploading the pixels row-by-row?
	for y, p := dr.Min.Y, 0; y < dr.Max.Y; y++ {
		t.w.glctx.TexSubImage2D(gl.TEXTURE_2D, 0, dr.Min.X, y, width, 1, gl.RGBA, gl.UNSIGNED_BYTE, pix[p:])
		p += buf.rgba.Stride
	}
}

func (t *textureImpl) Fill(dr image.Rectangle, src color.Color, op draw.Op) {
	t.w.glctxMu.Lock()
	defer t.w.glctxMu.Unlock()

	// TODO.
}

var quadCoords = f32Bytes(binary.LittleEndian,
	0, 0, // top left
	1, 0, // top right
	0, 1, // bottom left
	1, 1, // bottom right
)

const textureVertexSrc = `#version 100
uniform mat3 mvp;
uniform mat3 uvp;
attribute vec3 pos;
attribute vec2 inUV;
varying vec2 uv;
void main() {
	vec3 p = pos;
	p.z = 1.0;
	gl_Position = vec4(mvp * p, 1);
	uv = (uvp * vec3(inUV, 1)).xy;
}
`

const textureFragmentSrc = `#version 100
precision mediump float;
varying vec2 uv;
uniform sampler2D sample;
void main() {
	gl_FragColor = texture2D(sample, uv);
}
`

const fillVertexSrc = `#version 100
uniform mat3 mvp;
attribute vec3 pos;
void main() {
	vec3 p = pos;
	p.z = 1.0;
	gl_Position = vec4(mvp * p, 1);
}
`

const fillFragmentSrc = `#version 100
precision mediump float;
uniform vec4 color;
void main() {
	gl_FragColor = color;
}
`
