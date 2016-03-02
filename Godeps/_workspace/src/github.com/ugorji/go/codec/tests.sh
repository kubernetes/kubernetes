#!/bin/bash

# Run all the different permutations of all the tests.
# This helps ensure that nothing gets broken.

_run() {
    # 1. VARIATIONS: regular (t), canonical (c), IO R/W (i),
    #                binc-nosymbols (n), struct2array (s), intern string (e),
    #                json-indent (d), circular (l)
    # 2. MODE: reflection (r), external (x), codecgen (g), unsafe (u), notfastpath (f)
    # 3. OPTIONS: verbose (v), reset (z), must (m),
    # 
    # Use combinations of mode to get exactly what you want,
    # and then pass the variations you need.

    ztags=""
    zargs=""
    local OPTIND 
    OPTIND=1
    while getopts "_xurtcinsvgzmefdl" flag
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
            'xl') zargs="$zargs -tl" ;;
            *) ;;
        esac
    done
    # shift $((OPTIND-1))
    printf '............. TAGS: %s .............\n' "$ztags"
    # echo ">>>>>>> TAGS: $ztags"
    
    OPTIND=1
    while getopts "_xurtcinsvgzmefdl" flag
    do
        case "x$flag" in 
            'xt') printf ">>>>>>> REGULAR    : "; go test "-tags=$ztags" $zargs ; sleep 2 ;;
            'xc') printf ">>>>>>> CANONICAL  : "; go test "-tags=$ztags" $zargs -tc; sleep 2 ;;
            'xi') printf ">>>>>>> I/O        : "; go test "-tags=$ztags" $zargs -ti; sleep 2 ;;
            'xn') printf ">>>>>>> NO_SYMBOLS : "; go test "-tags=$ztags" -run=Binc $zargs -tn; sleep 2 ;;
            'xs') printf ">>>>>>> TO_ARRAY   : "; go test "-tags=$ztags" $zargs -ts; sleep 2 ;;
            'xe') printf ">>>>>>> INTERN     : "; go test "-tags=$ztags" $zargs -te; sleep 2 ;;
            'xd') printf ">>>>>>> INDENT     : ";
                  go test "-tags=$ztags" -run=JsonCodecsTable -td=-1 $zargs;
                  go test "-tags=$ztags" -run=JsonCodecsTable -td=8 $zargs;
                  sleep 2 ;;
            *) ;;
        esac
    done
    shift $((OPTIND-1))

    OPTIND=1
}

# echo ">>>>>>> RUNNING VARIATIONS OF TESTS"    
if [[ "x$@" = "x" ]]; then
    # All: r, x, g, gu
    _run "-_tcinsed_ml"  # regular
    _run "-_tcinsed_ml_z" # regular with reset
    _run "-_tcinsed_ml_f" # regular with no fastpath (notfastpath)
    _run "-x_tcinsed_ml" # external
    _run "-gx_tcinsed_ml" # codecgen: requires external
    _run "-gxu_tcinsed_ml" # codecgen + unsafe
elif [[ "x$@" = "x-Z" ]]; then
    # Regular
    _run "-_tcinsed_ml"  # regular
    _run "-_tcinsed_ml_z" # regular with reset
elif [[ "x$@" = "x-F" ]]; then
    # regular with notfastpath
    _run "-_tcinsed_ml_f"  # regular
    _run "-_tcinsed_ml_zf" # regular with reset
else
    _run "$@"
fi
