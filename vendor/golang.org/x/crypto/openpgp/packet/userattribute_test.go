// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package packet

import (
	"bytes"
	"encoding/base64"
	"image/color"
	"image/jpeg"
	"testing"
)

func TestParseUserAttribute(t *testing.T) {
	r := base64.NewDecoder(base64.StdEncoding, bytes.NewBufferString(userAttributePacket))
	for i := 0; i < 2; i++ {
		p, err := Read(r)
		if err != nil {
			t.Fatal(err)
		}
		uat := p.(*UserAttribute)
		imgs := uat.ImageData()
		if len(imgs) != 1 {
			t.Errorf("Unexpected number of images in user attribute packet: %d", len(imgs))
		}
		if len(imgs[0]) != 3395 {
			t.Errorf("Unexpected JPEG image size: %d", len(imgs[0]))
		}
		img, err := jpeg.Decode(bytes.NewBuffer(imgs[0]))
		if err != nil {
			t.Errorf("Error decoding JPEG image: %v", err)
		}
		// A pixel in my right eye.
		pixel := color.NRGBAModel.Convert(img.At(56, 36))
		ref := color.NRGBA{R: 157, G: 128, B: 124, A: 255}
		if pixel != ref {
			t.Errorf("Unexpected pixel color: %v", pixel)
		}
		w := bytes.NewBuffer(nil)
		err = uat.Serialize(w)
		if err != nil {
			t.Errorf("Error writing user attribute: %v", err)
		}
		r = bytes.NewBuffer(w.Bytes())
	}
}

const userAttributePacket = `
0cyWzJQBEAABAQAAAAAAAAAAAAAAAP/Y/+AAEEpGSUYAAQIAAAEAAQAA/9sAQwAFAwQEBAMFBAQE
BQUFBgcMCAcHBwcPCgsJDBEPEhIRDxEQExYcFxMUGhUQERghGBocHR8fHxMXIiQiHiQcHh8e/9sA
QwEFBQUHBgcOCAgOHhQRFB4eHh4eHh4eHh4eHh4eHh4eHh4eHh4eHh4eHh4eHh4eHh4eHh4eHh4e
Hh4eHh4eHh4e/8AAEQgAZABkAwEiAAIRAQMRAf/EAB8AAAEFAQEBAQEBAAAAAAAAAAABAgMEBQYH
CAkKC//EALUQAAIBAwMCBAMFBQQEAAABfQECAwAEEQUSITFBBhNRYQcicRQygZGhCCNCscEVUtHw
JDNicoIJChYXGBkaJSYnKCkqNDU2Nzg5OkNERUZHSElKU1RVVldYWVpjZGVmZ2hpanN0dXZ3eHl6
g4SFhoeIiYqSk5SVlpeYmZqio6Slpqeoqaqys7S1tre4ubrCw8TFxsfIycrS09TV1tfY2drh4uPk
5ebn6Onq8fLz9PX29/j5+v/EAB8BAAMBAQEBAQEBAQEAAAAAAAABAgMEBQYHCAkKC//EALURAAIB
AgQEAwQHBQQEAAECdwABAgMRBAUhMQYSQVEHYXETIjKBCBRCkaGxwQkjM1LwFWJy0QoWJDThJfEX
GBkaJicoKSo1Njc4OTpDREVGR0hJSlNUVVZXWFlaY2RlZmdoaWpzdHV2d3h5eoKDhIWGh4iJipKT
lJWWl5iZmqKjpKWmp6ipqrKztLW2t7i5usLDxMXGx8jJytLT1NXW19jZ2uLj5OXm5+jp6vLz9PX2
9/j5+v/aAAwDAQACEQMRAD8A5uGP06VehQ4pIox04q5EnHSvAep+hIIl4zVuMHGPWmRrUWtalaaN
pU2oXsgSGJSxPr6ClvoitErs0Itqjc7BQOpPAFYmrfEnwjojtHNqaXEynBjtx5hH4jj9a8B8d+Od
W8UXZjWR4LJT+7t0Jwfc+prnIdO1CWZEW2mZ3HyDactXXDB3V5s8evm1namj6r0H4weCLtxG+ova
ueP30RA/MV6not1bX0Ed1ZzxzwyDKvGwZSPqK+Ff+ES8R8t/ZV2oHUmM10Hgbxp4m8BatEfNnWBH
/eWshOxx9Kmpg4te49RUM1kn+8Wh9zQ4P1FaMC7l465rjPh14y0fxnoseoaXOpfaPOgJ+eI98j09
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`
