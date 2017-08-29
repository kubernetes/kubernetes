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
    # "_xurtcinsvgzmefdl" ===  "_cdefgilmnrtsuvxz"
    while getopts "_cdefgilmnrtsuvwxz" flag
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
            'xw') zargs="$zargs -tx=10" ;;
            *) ;;
        esac
    done
    # shift $((OPTIND-1))
    printf '............. TAGS: %s .............\n' "$ztags"
    # echo ">>>>>>> TAGS: $ztags"
    
    OPTIND=1
    while getopts "_cdefgilmnrtsuvwxz" flag
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
if [[ "x$@" = "x"  || "x$@" = "x-A" ]]; then
    # All: r, x, g, gu
    _run "-_tcinsed_ml"  # regular
    _run "-_tcinsed_ml_z" # regular with reset
    _run "-w_tcinsed_ml"  # regular with max init len
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
elif [[ "x$@" = "x-C" ]]; then
    # codecgen
    _run "-gx_tcinsed_ml" # codecgen: requires external
    _run "-gxu_tcinsed_ml" # codecgen + unsafe
    _run "-gxuw_tcinsed_ml" # codecgen + unsafe + maxinitlen
elif [[ "x$@" = "x-X" ]]; then
    # external
    _run "-x_tcinsed_ml" # external
elif [[ "x$@" = "x-h" || "x$@" = "x-?" ]]; then
    cat <<EOF
Usage: tests.sh [options...]
  -A run through all tests (regular, external, codecgen)
  -Z regular tests only 
  -F regular tests only (without fastpath, so they run quickly)
  -C codecgen only 
  -X external only 
  -h show help (usage)
  -? same as -h
  (no options) 
      same as -A
  (unrecognized options)
      just pass on the options from the command line 
EOF
else
    # e.g. ./tests.sh "-w_tcinsed_ml"
    _run "$@"
fi
