package antlr

type runtimeConfiguration struct {
	statsTraceStacks              bool
	lexerATNSimulatorDebug        bool
	lexerATNSimulatorDFADebug     bool
	parserATNSimulatorDebug       bool
	parserATNSimulatorTraceATNSim bool
	parserATNSimulatorDFADebug    bool
	parserATNSimulatorRetryDebug  bool
	lRLoopEntryBranchOpt          bool
	memoryManager                 bool
}

// Global runtime configuration
var runtimeConfig = runtimeConfiguration{
	lRLoopEntryBranchOpt: true,
}

type runtimeOption func(*runtimeConfiguration) error

// ConfigureRuntime allows the runtime to be configured globally setting things like trace and statistics options.
// It uses the functional options pattern for go. This is a package global function as it operates on the runtime
// configuration regardless of the instantiation of anything higher up such as a parser or lexer. Generally this is
// used for debugging/tracing/statistics options, which are usually used by the runtime maintainers (or rather the
// only maintainer). However, it is possible that you might want to use this to set a global option concerning the
// memory allocation type used by the runtime such as sync.Pool or not.
//
// The options are applied in the order they are passed in, so the last option will override any previous options.
//
// For example, if you want to turn on the collection create point stack flag to true, you can do:
//
//	antlr.ConfigureRuntime(antlr.WithStatsTraceStacks(true))
//
// If you want to turn it off, you can do:
//
//	antlr.ConfigureRuntime(antlr.WithStatsTraceStacks(false))
func ConfigureRuntime(options ...runtimeOption) error {
	for _, option := range options {
		err := option(&runtimeConfig)
		if err != nil {
			return err
		}
	}
	return nil
}

// WithStatsTraceStacks sets the global flag indicating whether to collect stack traces at the create-point of
// certain structs, such as collections, or the use point of certain methods such as Put().
// Because this can be expensive, it is turned off by default. However, it
// can be useful to track down exactly where memory is being created and used.
//
// Use:
//
//	antlr.ConfigureRuntime(antlr.WithStatsTraceStacks(true))
//
// You can turn it off at any time using:
//
//	antlr.ConfigureRuntime(antlr.WithStatsTraceStacks(false))
func WithStatsTraceStacks(trace bool) runtimeOption {
	return func(config *runtimeConfiguration) error {
		config.statsTraceStacks = trace
		return nil
	}
}

// WithLexerATNSimulatorDebug sets the global flag indicating whether to log debug information from the lexer [ATN]
// simulator. This is useful for debugging lexer issues by comparing the output with the Java runtime. Only useful
// to the runtime maintainers.
//
// Use:
//
//	antlr.ConfigureRuntime(antlr.WithLexerATNSimulatorDebug(true))
//
// You can turn it off at any time using:
//
//	antlr.ConfigureRuntime(antlr.WithLexerATNSimulatorDebug(false))
func WithLexerATNSimulatorDebug(debug bool) runtimeOption {
	return func(config *runtimeConfiguration) error {
		config.lexerATNSimulatorDebug = debug
		return nil
	}
}

// WithLexerATNSimulatorDFADebug sets the global flag indicating whether to log debug information from the lexer [ATN] [DFA]
// simulator. This is useful for debugging lexer issues by comparing the output with the Java runtime. Only useful
// to the runtime maintainers.
//
// Use:
//
//	antlr.ConfigureRuntime(antlr.WithLexerATNSimulatorDFADebug(true))
//
// You can turn it off at any time using:
//
//	antlr.ConfigureRuntime(antlr.WithLexerATNSimulatorDFADebug(false))
func WithLexerATNSimulatorDFADebug(debug bool) runtimeOption {
	return func(config *runtimeConfiguration) error {
		config.lexerATNSimulatorDFADebug = debug
		return nil
	}
}

// WithParserATNSimulatorDebug sets the global flag indicating whether to log debug information from the parser [ATN]
// simulator. This is useful for debugging parser issues by comparing the output with the Java runtime. Only useful
// to the runtime maintainers.
//
// Use:
//
//	antlr.ConfigureRuntime(antlr.WithParserATNSimulatorDebug(true))
//
// You can turn it off at any time using:
//
//	antlr.ConfigureRuntime(antlr.WithParserATNSimulatorDebug(false))
func WithParserATNSimulatorDebug(debug bool) runtimeOption {
	return func(config *runtimeConfiguration) error {
		config.parserATNSimulatorDebug = debug
		return nil
	}
}

// WithParserATNSimulatorTraceATNSim sets the global flag indicating whether to log trace information from the parser [ATN] simulator
// [DFA]. This is useful for debugging parser issues by comparing the output with the Java runtime. Only useful
// to the runtime maintainers.
//
// Use:
//
//	antlr.ConfigureRuntime(antlr.WithParserATNSimulatorTraceATNSim(true))
//
// You can turn it off at any time using:
//
//	antlr.ConfigureRuntime(antlr.WithParserATNSimulatorTraceATNSim(false))
func WithParserATNSimulatorTraceATNSim(trace bool) runtimeOption {
	return func(config *runtimeConfiguration) error {
		config.parserATNSimulatorTraceATNSim = trace
		return nil
	}
}

// WithParserATNSimulatorDFADebug sets the global flag indicating whether to log debug information from the parser [ATN] [DFA]
// simulator. This is useful for debugging parser issues by comparing the output with the Java runtime. Only useful
// to the runtime maintainers.
//
// Use:
//
//	antlr.ConfigureRuntime(antlr.WithParserATNSimulatorDFADebug(true))
//
// You can turn it off at any time using:
//
//	antlr.ConfigureRuntime(antlr.WithParserATNSimulatorDFADebug(false))
func WithParserATNSimulatorDFADebug(debug bool) runtimeOption {
	return func(config *runtimeConfiguration) error {
		config.parserATNSimulatorDFADebug = debug
		return nil
	}
}

// WithParserATNSimulatorRetryDebug sets the global flag indicating whether to log debug information from the parser [ATN] [DFA]
// simulator when retrying a decision. This is useful for debugging parser issues by comparing the output with the Java runtime.
// Only useful to the runtime maintainers.
//
// Use:
//
//	antlr.ConfigureRuntime(antlr.WithParserATNSimulatorRetryDebug(true))
//
// You can turn it off at any time using:
//
//	antlr.ConfigureRuntime(antlr.WithParserATNSimulatorRetryDebug(false))
func WithParserATNSimulatorRetryDebug(debug bool) runtimeOption {
	return func(config *runtimeConfiguration) error {
		config.parserATNSimulatorRetryDebug = debug
		return nil
	}
}

// WithLRLoopEntryBranchOpt sets the global flag indicating whether let recursive loop operations should be
// optimized or not. This is useful for debugging parser issues by comparing the output with the Java runtime.
// It turns off the functionality of [canDropLoopEntryEdgeInLeftRecursiveRule] in [ParserATNSimulator].
//
// Note that default is to use this optimization.
//
// Use:
//
//	antlr.ConfigureRuntime(antlr.WithLRLoopEntryBranchOpt(true))
//
// You can turn it off at any time using:
//
//	antlr.ConfigureRuntime(antlr.WithLRLoopEntryBranchOpt(false))
func WithLRLoopEntryBranchOpt(off bool) runtimeOption {
	return func(config *runtimeConfiguration) error {
		config.lRLoopEntryBranchOpt = off
		return nil
	}
}

// WithMemoryManager sets the global flag indicating whether to use the memory manager or not. This is useful
// for poorly constructed grammars that create a lot of garbage. It turns on the functionality of [memoryManager], which
// will intercept garbage collection and cause available memory to be reused. At the end of the day, this is no substitute
// for fixing your grammar by ridding yourself of extreme ambiguity. BUt if you are just trying to reuse an opensource
// grammar, this may help make it more practical.
//
// Note that default is to use normal Go memory allocation and not pool memory.
//
// Use:
//
//	antlr.ConfigureRuntime(antlr.WithMemoryManager(true))
//
// Note that if you turn this on, you should probably leave it on. You should use only one memory strategy or the other
// and should remember to nil out any references to the parser or lexer when you are done with them.
func WithMemoryManager(use bool) runtimeOption {
	return func(config *runtimeConfiguration) error {
		config.memoryManager = use
		return nil
	}
}
