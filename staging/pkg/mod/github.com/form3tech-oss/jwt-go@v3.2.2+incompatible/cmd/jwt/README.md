`jwt` command-line tool
=======================

This is a simple tool to sign, verify and show JSON Web Tokens from
the command line.

The following will create and sign a token, then verify it and output the original claims:

     echo {\"foo\":\"bar\"} | ./jwt -key ../../test/sample_key -alg RS256 -sign - | ./jwt -key ../../test/sample_key.pub -alg RS256 -verify -

Key files should be in PEM format. Other formats are not supported by this tool.

To simply display a token, use:

    echo $JWT | ./jwt -show -

You can install this tool with the following command:

     go install github.com/dgrijalva/jwt-go/cmd/jwt

