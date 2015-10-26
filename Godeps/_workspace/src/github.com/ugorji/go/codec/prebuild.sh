#!/bin/bash

# _needgen is a helper function to tell if we need to generate files for msgp, codecgen.
_needgen() {
    local a="$1"
    zneedgen=0
    if [[ ! -e "$a" ]]
    then
        zneedgen=1
        echo 1
        return 0
    fi 
    for i in `ls -1 *.go.tmpl gen.go values_test.go`
    do
        if [[ "$a" -ot "$i" ]]
        then
            zneedgen=1
            echo 1
            return 0
        fi 
    done 
    echo 0
}

# _build generates fast-path.go and gen-helper.go.
# 
# It is needed because there is some dependency between the generated code
# and the other classes. Consequently, we have to totally remove the 
# generated files and put stubs in place, before calling "go run" again
# to recreate them.
_build() {
    if ! [[ "${zforce}" == "1" ||
                "1" == $( _needgen "fast-path.generated.go" ) ||
                "1" == $( _needgen "gen-helper.generated.go" ) ||
                "1" == $( _needgen "gen.generated.go" ) ||
                1 == 0 ]]
    then
        return 0
    fi 

   # echo "Running prebuild"
    if [ "${zbak}" == "1" ] 
    then
        # echo "Backing up old generated files"
        _zts=`date '+%m%d%Y_%H%M%S'`
        _gg=".generated.go"
        [ -e "gen-helper${_gg}" ] && mv gen-helper${_gg} gen-helper${_gg}__${_zts}.bak
        [ -e "fast-path${_gg}" ] && mv fast-path${_gg} fast-path${_gg}__${_zts}.bak
        # [ -e "safe${_gg}" ] && mv safe${_gg} safe${_gg}__${_zts}.bak
        # [ -e "unsafe${_gg}" ] && mv unsafe${_gg} unsafe${_gg}__${_zts}.bak
    else 
        rm -f fast-path.generated.go gen.generated.go gen-helper.generated.go *safe.generated.go *_generated_test.go *.generated_ffjson_expose.go
    fi

    cat > gen.generated.go <<EOF
// Copyright (c) 2012-2015 Ugorji Nwoke. All rights reserved.
// Use of this source code is governed by a MIT license found in the LICENSE file.

package codec

// DO NOT EDIT. THIS FILE IS AUTO-GENERATED FROM gen-dec-(map|array).go.tmpl

const genDecMapTmpl = \`
EOF

    cat >> gen.generated.go < gen-dec-map.go.tmpl

    cat >> gen.generated.go <<EOF
\`

const genDecListTmpl = \`
EOF

    cat >> gen.generated.go < gen-dec-array.go.tmpl

    cat >> gen.generated.go <<EOF
\`

EOF
    # All functions, variables which must exist are put in this file.
    # This way, build works before we generate the right things.
    cat > fast-path.generated.go <<EOF
package codec 
import "reflect"
// func GenBytesToStringRO(b []byte) string { return string(b) }
func fastpathDecodeTypeSwitch(iv interface{}, d *Decoder) bool { return false }
func fastpathEncodeTypeSwitch(iv interface{}, e *Encoder) bool { return false }
func fastpathEncodeTypeSwitchSlice(iv interface{}, e *Encoder) bool { return false }
func fastpathEncodeTypeSwitchMap(iv interface{}, e *Encoder) bool { return false }
type fastpathE struct {
	rtid uintptr
	rt reflect.Type 
	encfn func(*encFnInfo, reflect.Value)
	decfn func(*decFnInfo, reflect.Value)
}
type fastpathA [0]fastpathE
func (x fastpathA) index(rtid uintptr) int { return -1 }
var fastpathAV fastpathA 

EOF

    cat > gen-from-tmpl.codec.generated.go <<EOF
package codec 
import "io"
func GenInternalGoFile(r io.Reader, w io.Writer, safe bool) error {
return genInternalGoFile(r, w, safe)
}
EOF
    
    cat > gen-from-tmpl.generated.go <<EOF
//+build ignore

package main

//import "flag"
import "ugorji.net/codec"
import "os"

func run(fnameIn, fnameOut string, safe bool) {
fin, err := os.Open(fnameIn)
if err != nil { panic(err) }
defer fin.Close()
fout, err := os.Create(fnameOut)
if err != nil { panic(err) }
defer fout.Close()
err = codec.GenInternalGoFile(fin, fout, safe)
if err != nil { panic(err) }
}

func main() {
// do not make safe/unsafe variants. 
// Instead, depend on escape analysis, and place string creation and usage appropriately.
// run("unsafe.go.tmpl", "safe.generated.go", true)
// run("unsafe.go.tmpl", "unsafe.generated.go", false)
run("fast-path.go.tmpl", "fast-path.generated.go", false)
run("gen-helper.go.tmpl", "gen-helper.generated.go", false)
}

EOF
    go run gen-from-tmpl.generated.go && \
        rm -f gen-from-tmpl.*generated.go 
}

_codegenerators() {
    if [[ $zforce == "1" || 
                "1" == $( _needgen "values_codecgen${zsfx}" ) ||
                "1" == $( _needgen "values_msgp${zsfx}" ) ||
                "1" == $( _needgen "values_ffjson${zsfx}" ) ||
                1 == 0 ]] 
    then
        # codecgen creates some temporary files in the directory (main, pkg).
        # Consequently, we should start msgp and ffjson first, and also put a small time latency before
        # starting codecgen.
        # Without this, ffjson chokes on one of the temporary files from codecgen.
        echo "ffjson ... " && \
            ffjson -w values_ffjson${zsfx} $zfin &
        zzzIdFF=$!
        echo "msgp ... " && \
            msgp -tests=false -pkg=codec -o=values_msgp${zsfx} -file=$zfin &
        zzzIdMsgp=$!

        sleep 1 # give ffjson and msgp some buffer time. see note above.

        echo "codecgen - !unsafe ... " && \
            codecgen -rt codecgen -t 'x,codecgen,!unsafe' -o values_codecgen${zsfx} -d 19780 $zfin &
        zzzIdC=$!
        echo "codecgen - unsafe ... " && \
            codecgen  -u -rt codecgen -t 'x,codecgen,unsafe' -o values_codecgen_unsafe${zsfx} -d 19781 $zfin &
        zzzIdCU=$!
        wait $zzzIdC $zzzIdCU $zzzIdMsgp $zzzIdFF && \
            # remove (M|Unm)arshalJSON implementations, so they don't conflict with encoding/json bench \
            sed -i 's+ MarshalJSON(+ _MarshalJSON(+g' values_ffjson${zsfx} && \
            sed -i 's+ UnmarshalJSON(+ _UnmarshalJSON(+g' values_ffjson${zsfx} && \
            echo "generators done!" && \
            true
    fi 
}

# _init reads the arguments and sets up the flags
_init() {
OPTIND=1
while getopts "fb" flag
do
    case "x$flag" in 
        'xf') zforce=1;;
        'xb') zbak=1;;
        *) echo "prebuild.sh accepts [-fb] only"; return 1;;
    esac
done
shift $((OPTIND-1))
OPTIND=1
}

# main script.
# First ensure that this is being run from the basedir (i.e. dirname of script is .)
if [ "." = `dirname $0` ]
then
    zmydir=`pwd`
    zfin="test_values.generated.go"
    zsfx="_generated_test.go"
    # rm -f *_generated_test.go 
    rm -f codecgen-*.go && \
        _init "$@" && \
        _build && \
        cp $zmydir/values_test.go $zmydir/$zfin && \
        _codegenerators && \
        echo prebuild done successfully
    rm -f $zmydir/$zfin
else
    echo "Script must be run from the directory it resides in"
fi 

