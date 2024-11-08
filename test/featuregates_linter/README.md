This directory contains static analysis scripts for verify functions.

Currently, the following commands are implemented:
```
go run test/featuregates_linter/main.go feature-gates verify-no-new-unversioned --new-features-file="${new_features_file}" --old-features-file="${old_features_file}"

go run test/featuregates_linter/main.go feature-gates verify-alphabetic-order --features-file="${features_file}"
```
