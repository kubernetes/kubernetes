This package runs a set of the Wycheproof tests provided by
https://github.com/google/wycheproof.

The JSON test files live in
https://github.com/google/wycheproof/tree/master/testvectors
and are being fetched and cached at a pinned version every time
these tests are run. To change the version of the wycheproof
repository that is being used for testing, update wycheproofModVer.

The structs for these tests are generated from the
schemas provided in https://github.com/google/wycheproof/tree/master/schemas
using https://github.com/a-h/generate.