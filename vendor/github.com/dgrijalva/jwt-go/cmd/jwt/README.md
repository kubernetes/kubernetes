`jwt` command-line tool
=======================

This is a simple tool to sign, verify and show JSON Web Tokens from
the command line.

The following will create and sign a token, then verify it and output the original claims:

     echo {\"foo\":\"bar\"} | bin/jwt -key test/sample_key -alg RS256 -sign - | bin/jwt -key test/sample_key.pub -verify -

To simply display a token, use:

    echo $JWT | jwt -show -
