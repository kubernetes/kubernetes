/*
   Copyright The containerd Authors.

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
*/

package log

import (
	"io"
	"log/slog"
	"sync"

	"github.com/sirupsen/logrus"
	lslog "github.com/sirupsen/logrus/hooks/slog"
)

// slogOut is used to set the slog logger when setting output format.
var slogOut io.Writer

// slogOnce guards UseSlog so repeated calls do not stack up hooks or
// reset slogOut to the discard writer installed on the first call.
var slogOnce sync.Once

func UseSlog() {
	slogOnce.Do(func() {
		L.Logger.SetNoLock()
		L.Logger.AddHook(slogHook{})
		slogOut = L.Logger.Out

		// Disable logrus formatting and output, as both are handled by slog.
		L.Logger.SetFormatter(discardFormatter{})
		L.Logger.SetOutput(io.Discard)
	})
}

type slogHook struct{}

func (slogHook) Levels() []logrus.Level {
	return logrus.AllLevels
}

func (slogHook) Fire(entry *logrus.Entry) error {
	return lslog.NewHook(slog.Default(), nil).Fire(entry)
}

// loggerLevel exposes the current Logrus logger level as a slog.Leveler.
type loggerLevel struct{}

func (loggerLevel) Level() slog.Level {
	return lslog.Level(L.Logger.GetLevel()).Level()
}

type discardFormatter struct{}

func (discardFormatter) Format(*logrus.Entry) ([]byte, error) { return nil, nil }
