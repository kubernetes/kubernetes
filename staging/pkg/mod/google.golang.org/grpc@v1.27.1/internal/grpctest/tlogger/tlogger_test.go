/*
 *
 * Copyright 2020 gRPC authors.
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
 *
 */

package tlogger

import (
	"reflect"
	"runtime"
	"strings"
	"testing"

	"google.golang.org/grpc/grpclog"
)

func TestInfo(t *testing.T) {
	Update(t)
	grpclog.Info("Info", "message.")
}

func TestInfoln(t *testing.T) {
	Update(t)
	grpclog.Infoln("Info", "message.")
}

func TestInfof(t *testing.T) {
	Update(t)
	grpclog.Infof("%v %v.", "Info", "message")
}

func TestWarning(t *testing.T) {
	Update(t)
	grpclog.Warning("Warning", "message.")
}

func TestWarningln(t *testing.T) {
	Update(t)
	grpclog.Warningln("Warning", "message.")
}

func TestWarningf(t *testing.T) {
	Update(t)
	grpclog.Warningf("%v %v.", "Warning", "message")
}

func TestSubTests(t *testing.T) {
	testFuncs := [6]func(*testing.T){TestInfo, TestInfoln, TestInfof, TestWarning, TestWarningln, TestWarningf}
	for _, testFunc := range testFuncs {
		splitFuncName := strings.Split(runtime.FuncForPC(reflect.ValueOf(testFunc).Pointer()).Name(), ".")
		t.Run(splitFuncName[len(splitFuncName)-1], testFunc)
	}
}
