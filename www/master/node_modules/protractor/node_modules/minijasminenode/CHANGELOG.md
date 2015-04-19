# 1.1.1
## Fixes
- ([a75f489](https://github.com/juliemr/minijasminenode/commit/a75f4894852e8cbb3a9b3a556ff993dced200cea)) 
  fix(reporter): report full date as well as relative

# 1.1.0

## Features
- ([665ea73](https://github.com/juliemr/minijasminenode/commit/665ea73352eb92c98706da1d9f008eefd6bb89e0)) 
  feat(reporter): add the realtimeFailure option, to report failures as they occur

  Instead of aggregating the output until the end. Use `--realtimeFailure` from the command line or
  set `options.realtimeFailure = true`.

  Also cleanup unused functions in the runner.

  See #19

- ([0821f58](https://github.com/juliemr/minijasminenode/commit/0821f58eabbe81b298ad87487901304fec384c87)) 
  feat(options): add a silent option which outputs nothing to the console

  Use --silent on the command line or add `silent: true` to the options.

- ([eb4c6e2](https://github.com/juliemr/minijasminenode/commit/eb4c6e24814dd902c23441458b42f9565b0717c1)) 
  feat(reporter): verbose reporter should print pass or fail after spec names

  See #21.

- ([a1a22df](https://github.com/juliemr/minijasminenode/commit/a1a22df6c4dec39be5398e1373a5a4cc81c2a073)) 
  feat(reporter): add timing information with the showTiming option and make verbose output in real
  time
