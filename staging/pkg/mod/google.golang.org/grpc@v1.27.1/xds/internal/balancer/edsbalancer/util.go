/*
 * Copyright 2019 gRPC authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package edsbalancer

import "google.golang.org/grpc/internal/wrr"

type dropper struct {
	// Drop rate will be numerator/denominator.
	numerator   uint32
	denominator uint32
	w           wrr.WRR
	category    string
}

func newDropper(numerator, denominator uint32, category string) *dropper {
	w := newRandomWRR()
	w.Add(true, int64(numerator))
	w.Add(false, int64(denominator-numerator))

	return &dropper{
		numerator:   numerator,
		denominator: denominator,
		w:           w,
		category:    category,
	}
}

func (d *dropper) drop() (ret bool) {
	return d.w.Next().(bool)
}
