#!/bin/bash

# Run all the different permutations of all the tests.
# This helps ensure that nothing gets broken.

_run() {
    # 1. VARIATIONS: regular (t), canonical (c), IO R/W (i),
    #                binc-nosymbols (n), struct2array (s), intern string (e),
    # 2. MODE: reflection (r), external (x), codecgen (g), unsafe (u), notfastpath (f)
    # 3. OPTIONS: verbose (v), reset (z), must (m),
    # 
    # Use combinations of mode to get exactly what you want,
    # and then pass the variations you need.

    ztags=""
    zargs=""
    local OPTIND 
    OPTIND=1
    while getopts "xurtcinsvgzmef" flag
    do
        case "x$flag" in 
            'xr')  ;;
            'xf') ztags="$ztags notfastpath" ;;
            'xg') ztags="$ztags codecgen" ;;
            'xx') ztags="$ztags x" ;;
            'xu') ztags="$ztags unsafe" ;;
            'xv') zargs="$zargs -tv" ;;
            'xz') zargs="$zargs -tr" ;;
            'xm') zargs="$zargs -tm" ;;
            *) ;;
        esac
    done
    # shift $((OPTIND-1))
    printf '............. TAGS: %s .............\n' "$ztags"
    # echo ">>>>>>> TAGS: $ztags"
    
    OPTIND=1
    while getopts "xurtcinsvgzmef" flag
    do
        case "x$flag" in 
            'xt') printf ">>>>>>> REGULAR    : "; go test "-tags=$ztags" $zargs ; sleep 2 ;;
            'xc') printf ">>>>>>> CANONICAL  : "; go test "-tags=$ztags" $zargs -tc; sleep 2 ;;
            'xi') printf ">>>>>>> I/O        : "; go test "-tags=$ztags" $zargs -ti; sleep 2 ;;
            'xn') printf ">>>>>>> NO_SYMBOLS : "; go test "-tags=$ztags" $zargs -tn; sleep 2 ;;
            'xs') printf ">>>>>>> TO_ARRAY   : "; go test "-tags=$ztags" $zargs -ts; sleep 2 ;;
            'xe') printf ">>>>>>> INTERN     : "; go test "-tags=$ztags" $zargs -te; sleep 2 ;;
            *) ;;
        esac
    done
    shift $((OPTIND-1))

    OPTIND=1
}

# echo ">>>>>>> RUNNING VARIATIONS OF TESTS"    
if [[ "x$@" = "x" ]]; then
    # All: r, x, g, gu
    _run "-rtcinsm"  # regular
    _run "-rtcinsmz" # regular with reset
    _run "-rtcinsmf" # regular with no fastpath (notfastpath)
    _run "-xtcinsm" # external
    _run "-gxtcinsm" # codecgen: requires external
    _run "-gxutcinsm" # codecgen + unsafe
elif [[ "x$@" = "x-Z" ]]; then
    # Regular
    _run "-rtcinsm"  # regular
    _run "-rtcinsmz" # regular with reset
elif [[ "x$@" = "x-F" ]]; then
    # regular with notfastpath
    _run "-rtcinsmf"  # regular
    _run "-rtcinsmzf" # regular with reset
else
    _run "$@"
fi
