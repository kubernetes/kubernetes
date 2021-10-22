# Copyright 2020 The Go Authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

# This script is executed by OSS-Fuzz's build to create fuzzer binaries.

function compile_fuzzer {
  path=$1
  function=$2
  fuzzer=$3

   # Instrument all Go files relevant to this fuzzer
  go-fuzz -tags=protolegacy -func $function -o $fuzzer.a $path

   # Instrumented, compiled Go ($fuzzer.a) + fuzzing engine = fuzzer binary
  $CXX $CXXFLAGS $LIB_FUZZING_ENGINE $fuzzer.a -lpthread -o $OUT/$fuzzer
}

for x in internal/fuzz/*; do
  if [ -d $x/corpus ]; then
    name=$(basename $x)
    compile_fuzzer google.golang.org/protobuf/$x Fuzz $name
    zip -jr $OUT/${name}_seed_corpus.zip $x/corpus
  fi
done
