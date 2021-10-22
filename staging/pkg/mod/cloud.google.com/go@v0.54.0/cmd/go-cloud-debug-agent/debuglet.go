// Copyright 2016 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// +build linux,go1.7

package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"io/ioutil"
	"log"
	"math/rand"
	"os"
	"sync"
	"time"

	"cloud.google.com/go/cmd/go-cloud-debug-agent/internal/breakpoints"
	debuglet "cloud.google.com/go/cmd/go-cloud-debug-agent/internal/controller"
	"cloud.google.com/go/cmd/go-cloud-debug-agent/internal/debug"
	"cloud.google.com/go/cmd/go-cloud-debug-agent/internal/debug/local"
	"cloud.google.com/go/cmd/go-cloud-debug-agent/internal/valuecollector"
	"cloud.google.com/go/compute/metadata"
	"golang.org/x/oauth2"
	"golang.org/x/oauth2/google"
	cd "google.golang.org/api/clouddebugger/v2"
)

var (
	appModule         = flag.String("appmodule", "", "Optional application module name.")
	appVersion        = flag.String("appversion", "", "Optional application module version name.")
	sourceContextFile = flag.String("sourcecontext", "", "File containing JSON-encoded source context.")
	verbose           = flag.Bool("v", false, "Output verbose log messages.")
	projectNumber     = flag.String("projectnumber", "", "Project number."+
		"  If this is not set, it is read from the GCP metadata server.")
	projectID = flag.String("projectid", "", "Project ID."+
		"  If this is not set, it is read from the GCP metadata server.")
	serviceAccountFile = flag.String("serviceaccountfile", "", "File containing JSON service account credentials.")
)

const (
	maxCapturedStackFrames = 50
	maxCapturedVariables   = 1000
)

func main() {
	flag.Usage = usage
	flag.Parse()
	args := flag.Args()
	if len(args) == 0 {
		// The user needs to supply the name of the executable to run.
		flag.Usage()
		return
	}
	if *projectNumber == "" {
		var err error
		*projectNumber, err = metadata.NumericProjectID()
		if err != nil {
			log.Print("Debuglet initialization: ", err)
		}
	}
	if *projectID == "" {
		var err error
		*projectID, err = metadata.ProjectID()
		if err != nil {
			log.Print("Debuglet initialization: ", err)
		}
	}
	sourceContexts, err := readSourceContextFile(*sourceContextFile)
	if err != nil {
		log.Print("Reading source context file: ", err)
	}
	var ts oauth2.TokenSource
	ctx := context.Background()
	if *serviceAccountFile != "" {
		if ts, err = serviceAcctTokenSource(ctx, *serviceAccountFile, cd.CloudDebuggerScope); err != nil {
			log.Fatalf("Error getting credentials from file %s: %v", *serviceAccountFile, err)
		}
	} else if ts, err = google.DefaultTokenSource(ctx, cd.CloudDebuggerScope); err != nil {
		log.Print("Error getting application default credentials for Cloud Debugger:", err)
		os.Exit(103)
	}
	c, err := debuglet.NewController(ctx, debuglet.Options{
		ProjectNumber:  *projectNumber,
		ProjectID:      *projectID,
		AppModule:      *appModule,
		AppVersion:     *appVersion,
		SourceContexts: sourceContexts,
		Verbose:        *verbose,
		TokenSource:    ts,
	})
	if err != nil {
		log.Fatal("Error connecting to Cloud Debugger: ", err)
	}
	prog, err := local.New(args[0])
	if err != nil {
		log.Fatal("Error loading program: ", err)
	}
	// Load the program, but don't actually start it running yet.
	if _, err = prog.Run(args[1:]...); err != nil {
		log.Fatal("Error loading program: ", err)
	}
	bs := breakpoints.NewBreakpointStore(prog)

	// Seed the random number generator.
	rand.Seed(time.Now().UnixNano())

	// Now we want to do two things: run the user's program, and start sending
	// List requests periodically to the Debuglet Controller to get breakpoints
	// to set.
	//
	// We want to give the Debuglet Controller a chance to give us breakpoints
	// before we start the program, otherwise we would miss any breakpoint
	// triggers that occur during program startup -- for example, a breakpoint on
	// the first line of main. But if the Debuglet Controller is not responding or
	// is returning errors, we don't want to delay starting the program
	// indefinitely.
	//
	// We pass a channel to breakpointListLoop, which will close it when the first
	// List call finishes.  Then we wait until either the channel is closed or a
	// 5-second timer has finished before starting the program.
	ch := make(chan bool)
	// Start a goroutine that sends List requests to the Debuglet Controller, and
	// sets any breakpoints it gets back.
	go breakpointListLoop(ctx, c, bs, ch)
	// Wait until 5 seconds have passed or breakpointListLoop has closed ch.
	select {
	case <-time.After(5 * time.Second):
	case <-ch:
	}
	// Run the debuggee.
	programLoop(ctx, c, bs, prog)
}

// usage prints a usage message to stderr and exits.
func usage() {
	me := "a.out"
	if len(os.Args) >= 1 {
		me = os.Args[0]
	}
	fmt.Fprintf(os.Stderr, "Usage of %s:\n", me)
	fmt.Fprintf(os.Stderr, "\t%s [flags...] -- <program name> args...\n", me)
	fmt.Fprintf(os.Stderr, "Flags:\n")
	flag.PrintDefaults()
	fmt.Fprintf(os.Stderr,
		"See https://cloud.google.com/tools/cloud-debugger/setting-up-on-compute-engine for more information.\n")
	os.Exit(2)
}

// readSourceContextFile reads a JSON-encoded source context from the given file.
// It returns a non-empty slice on success.
func readSourceContextFile(filename string) ([]*cd.SourceContext, error) {
	if filename == "" {
		return nil, nil
	}
	scJSON, err := ioutil.ReadFile(filename)
	if err != nil {
		return nil, fmt.Errorf("reading file %q: %v", filename, err)
	}
	var sc cd.SourceContext
	if err = json.Unmarshal(scJSON, &sc); err != nil {
		return nil, fmt.Errorf("parsing file %q: %v", filename, err)
	}
	return []*cd.SourceContext{&sc}, nil
}

// breakpointListLoop repeatedly calls the Debuglet Controller's List RPC, and
// passes the results to the BreakpointStore so it can set and unset breakpoints
// in the program.
//
// After the first List call finishes, ch is closed.
func breakpointListLoop(ctx context.Context, c *debuglet.Controller, bs *breakpoints.BreakpointStore, first chan bool) {
	const (
		avgTimeBetweenCalls = time.Second
		errorDelay          = 5 * time.Second
	)

	// randomDuration returns a random duration with expected value avg.
	randomDuration := func(avg time.Duration) time.Duration {
		return time.Duration(rand.Int63n(int64(2*avg + 1)))
	}

	var consecutiveFailures uint

	for {
		callStart := time.Now()
		resp, err := c.List(ctx)
		if err != nil && err != debuglet.ErrListUnchanged {
			log.Printf("Debuglet controller server error: %v", err)
		}
		if err == nil {
			bs.ProcessBreakpointList(resp.Breakpoints)
		}

		if first != nil {
			// We've finished one call to List and set any breakpoints we received.
			close(first)
			first = nil
		}

		// Asynchronously send updates for any breakpoints that caused an error when
		// the BreakpointStore tried to process them.  We don't wait for the update
		// to finish before the program can exit, as we do for normal updates.
		errorBps := bs.ErrorBreakpoints()
		for _, bp := range errorBps {
			go func(bp *cd.Breakpoint) {
				if err := c.Update(ctx, bp.Id, bp); err != nil {
					log.Printf("Failed to send breakpoint update for %s: %s", bp.Id, err)
				}
			}(bp)
		}

		// Make the next call not too soon after the one we just did.
		delay := randomDuration(avgTimeBetweenCalls)

		// If the call returned an error other than ErrListUnchanged, wait longer.
		if err != nil && err != debuglet.ErrListUnchanged {
			// Wait twice as long after each consecutive failure, to a maximum of 16x.
			delay += randomDuration(errorDelay * (1 << consecutiveFailures))
			if consecutiveFailures < 4 {
				consecutiveFailures++
			}
		} else {
			consecutiveFailures = 0
		}

		// Sleep until we reach time callStart+delay.  If we've already passed that
		// time, time.Sleep will return immediately -- this should be the common
		// case, since the server will delay responding to List for a while when
		// there are no changes to report.
		time.Sleep(callStart.Add(delay).Sub(time.Now()))
	}
}

// programLoop runs the program being debugged to completion.  When a breakpoint's
// conditions are satisfied, it sends an Update RPC to the Debuglet Controller.
// The function returns when the program exits and all Update RPCs have finished.
func programLoop(ctx context.Context, c *debuglet.Controller, bs *breakpoints.BreakpointStore, prog debug.Program) {
	var wg sync.WaitGroup
	for {
		// Run the program until it hits a breakpoint or exits.
		status, err := prog.Resume()
		if err != nil {
			break
		}

		// Get the breakpoints at this address whose conditions were satisfied,
		// and remove the ones that aren't logpoints.
		bps := bs.BreakpointsAtPC(status.PC)
		bps = bpsWithConditionSatisfied(bps, prog)
		for _, bp := range bps {
			if bp.Action != "LOG" {
				bs.RemoveBreakpoint(bp)
			}
		}

		if len(bps) == 0 {
			continue
		}

		// Evaluate expressions and get the stack.
		vc := valuecollector.NewCollector(prog, maxCapturedVariables)
		needStackFrames := false
		for _, bp := range bps {
			// If evaluating bp's condition didn't return an error, evaluate bp's
			// expressions, and later get the stack frames.
			if bp.Status == nil {
				bp.EvaluatedExpressions = expressionValues(bp.Expressions, prog, vc)
				needStackFrames = true
			}
		}
		var (
			stack                    []*cd.StackFrame
			stackFramesStatusMessage *cd.StatusMessage
		)
		if needStackFrames {
			stack, stackFramesStatusMessage = stackFrames(prog, vc)
		}

		// Read variable values from the program.
		variableTable := vc.ReadValues()

		// Start a goroutine to send updates to the Debuglet Controller or write
		// to logs, concurrently with resuming the program.
		// TODO: retry Update on failure.
		for _, bp := range bps {
			wg.Add(1)
			switch bp.Action {
			case "LOG":
				go func(format string, evaluatedExpressions []*cd.Variable) {
					s := valuecollector.LogString(format, evaluatedExpressions, variableTable)
					log.Print(s)
					wg.Done()
				}(bp.LogMessageFormat, bp.EvaluatedExpressions)
				bp.Status = nil
				bp.EvaluatedExpressions = nil
			default:
				go func(bp *cd.Breakpoint) {
					defer wg.Done()
					bp.IsFinalState = true
					if bp.Status == nil {
						// If evaluating bp's condition didn't return an error, include the
						// stack frames, variable table, and any status message produced when
						// getting the stack frames.
						bp.StackFrames = stack
						bp.VariableTable = variableTable
						bp.Status = stackFramesStatusMessage
					}
					if err := c.Update(ctx, bp.Id, bp); err != nil {
						log.Printf("Failed to send breakpoint update for %s: %s", bp.Id, err)
					}
				}(bp)
			}
		}
	}

	// Wait for all updates to finish before returning.
	wg.Wait()
}

// bpsWithConditionSatisfied returns the breakpoints whose conditions are true
// (or that do not have a condition.)
func bpsWithConditionSatisfied(bpsIn []*cd.Breakpoint, prog debug.Program) []*cd.Breakpoint {
	var bpsOut []*cd.Breakpoint
	for _, bp := range bpsIn {
		cond, err := condTruth(bp.Condition, prog)
		if err != nil {
			bp.Status = errorStatusMessage(err.Error(), refersToBreakpointCondition)
			// Include bp in the list to be updated when there's an error, so that
			// the user gets a response.
			bpsOut = append(bpsOut, bp)
		} else if cond {
			bpsOut = append(bpsOut, bp)
		}
	}
	return bpsOut
}

// condTruth evaluates a condition.
func condTruth(condition string, prog debug.Program) (bool, error) {
	if condition == "" {
		// A condition wasn't set.
		return true, nil
	}
	val, err := prog.Evaluate(condition)
	if err != nil {
		return false, err
	}
	if v, ok := val.(bool); !ok {
		return false, fmt.Errorf("condition expression has type %T, should be bool", val)
	} else {
		return v, nil
	}
}

// expressionValues evaluates a slice of expressions and returns a []*cd.Variable
// containing the results.
// If the result of an expression evaluation refers to values from the program's
// memory (e.g., the expression evaluates to a slice) a corresponding variable is
// added to the value collector, to be read later.
func expressionValues(expressions []string, prog debug.Program, vc *valuecollector.Collector) []*cd.Variable {
	evaluatedExpressions := make([]*cd.Variable, len(expressions))
	for i, exp := range expressions {
		ee := &cd.Variable{Name: exp}
		evaluatedExpressions[i] = ee
		if val, err := prog.Evaluate(exp); err != nil {
			ee.Status = errorStatusMessage(err.Error(), refersToBreakpointExpression)
		} else {
			vc.FillValue(val, ee)
		}
	}
	return evaluatedExpressions
}

// stackFrames returns a stack trace for the program.  It passes references to
// function parameters and local variables to the value collector, so it can read
// their values later.
func stackFrames(prog debug.Program, vc *valuecollector.Collector) ([]*cd.StackFrame, *cd.StatusMessage) {
	frames, err := prog.Frames(maxCapturedStackFrames)
	if err != nil {
		return nil, errorStatusMessage("Error getting stack: "+err.Error(), refersToUnspecified)
	}
	stackFrames := make([]*cd.StackFrame, len(frames))
	for i, f := range frames {
		frame := &cd.StackFrame{}
		frame.Function = f.Function
		for _, v := range f.Params {
			frame.Arguments = append(frame.Arguments, vc.AddVariable(debug.LocalVar(v)))
		}
		for _, v := range f.Vars {
			frame.Locals = append(frame.Locals, vc.AddVariable(v))
		}
		frame.Location = &cd.SourceLocation{
			Path: f.File,
			Line: int64(f.Line),
		}
		stackFrames[i] = frame
	}
	return stackFrames, nil
}

// errorStatusMessage returns a *cd.StatusMessage indicating an error,
// with the given message and refersTo field.
func errorStatusMessage(msg string, refersTo int) *cd.StatusMessage {
	return &cd.StatusMessage{
		Description: &cd.FormatMessage{Format: "$0", Parameters: []string{msg}},
		IsError:     true,
		RefersTo:    refersToString[refersTo],
	}
}

const (
	// RefersTo values for cd.StatusMessage.
	refersToUnspecified = iota
	refersToBreakpointCondition
	refersToBreakpointExpression
)

// refersToString contains the strings for each refersTo value.
// See the definition of StatusMessage in the v2/clouddebugger package.
var refersToString = map[int]string{
	refersToUnspecified:          "UNSPECIFIED",
	refersToBreakpointCondition:  "BREAKPOINT_CONDITION",
	refersToBreakpointExpression: "BREAKPOINT_EXPRESSION",
}

func serviceAcctTokenSource(ctx context.Context, filename string, scope ...string) (oauth2.TokenSource, error) {
	data, err := ioutil.ReadFile(filename)
	if err != nil {
		return nil, fmt.Errorf("cannot read service account file: %v", err)
	}
	cfg, err := google.JWTConfigFromJSON(data, scope...)
	if err != nil {
		return nil, fmt.Errorf("google.JWTConfigFromJSON: %v", err)
	}
	return cfg.TokenSource(ctx), nil
}
