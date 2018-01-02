// Copyright 2016 Unknwon
//
// Licensed under the Apache License, Version 2.0 (the "License"): you may
// not use this file except in compliance with the License. You may obtain
// a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
// WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
// License for the specific language governing permissions and limitations
// under the License.

package ini

import (
	"testing"

	. "github.com/smartystreets/goconvey/convey"
)

func Test_BOM(t *testing.T) {
	Convey("Test handling BOM", t, func() {
		Convey("UTF-8-BOM", func() {
			cfg, err := Load("testdata/UTF-8-BOM.ini")
			So(err, ShouldBeNil)
			So(cfg, ShouldNotBeNil)

			So(cfg.Section("author").Key("E-MAIL").String(), ShouldEqual, "u@gogs.io")
		})

		Convey("UTF-16-LE-BOM", func() {
			cfg, err := Load("testdata/UTF-16-LE-BOM.ini")
			So(err, ShouldBeNil)
			So(cfg, ShouldNotBeNil)
		})

		Convey("UTF-16-BE-BOM", func() {
		})
	})
}
