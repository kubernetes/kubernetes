#!/bin/bash

# Run all the different permutations of all the tests.
# This helps ensure that nothing gets broken.

_run() {
    # 1. VARIATIONS: regular (t), canonical (c), IO R/W (i), binc-nosymbols (n), struct2array (s)
    # 2. MODE: reflection (r), codecgen (x), codecgen+unsafe (u)
    # 
    # Typically, you would run a combination of one value from a and b.

    ztags=""
    local OPTIND 
    OPTIND=1
    while getopts "xurtcinsvg" flag
    do
        case "x$flag" in 
            'xr')  ;;
            'xg') ztags="$ztags codecgen" ;;
            'xx') ztags="$ztags x" ;;
            'xu') ztags="$ztags unsafe" ;;
            'xv') zverbose="-tv" ;; 
            *) ;;
        esac
    done
    # shift $((OPTIND-1))
    printf '............. TAGS: %s .............\n' "$ztags"
    # echo ">>>>>>> TAGS: $ztags"
    
    OPTIND=1
    while getopts "xurtcinsvg" flag
    do
        case "x$flag" in 
            'xt') printf ">>>>>>> REGULAR    : "; go test "-tags=$ztags" "$zverbose" ; sleep 2 ;;
            'xc') printf ">>>>>>> CANONICAL  : "; go test "-tags=$ztags" "$zverbose" -tc; sleep 2 ;;
            'xi') printf ">>>>>>> I/O        : "; go test "-tags=$ztags" "$zverbose" -ti; sleep 2 ;;
            'xn') printf ">>>>>>> NO_SYMBOLS : "; go test "-tags=$ztags" "$zverbose" -tn; sleep 2 ;;
            'xs') printf ">>>>>>> TO_ARRAY   : "; go test "-tags=$ztags" "$zverbose" -ts; sleep 2 ;;
            *) ;;
        esac
    done
    shift $((OPTIND-1))

    OPTIND=1
}

# echo ">>>>>>> RUNNING VARIATIONS OF TESTS"    
if [[ "x$@" = "x" ]]; then
    # r, x, g, gu
    _run "-rtcins"
    _run "-xtcins"
    _run "-gtcins"
    _run "-gutcins"
else
    _run "$@"
fi
