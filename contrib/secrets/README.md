Secrets generation made easy
============================

Generating filesystem secrets is a headache at the moment, so this python 
script will help doing it. In your k8s config folder, create a folder for
your secrets, and put this script inside.

DISCLAIMER: Use python3

The use is quite simple. For each secret you want, create a folder with the
metadata name of the secret, and fill the folder with all the files you want
to generate. Remember that the files should be valid DNS names.

The script will recurse each folder, and generate a json file with the
filenames as keys and their content in base64.

example layout:
```console
secrets/
  nginxfiles/
    default.conf
    ssl.crt
    ssl.key
```

The output would be a json file `secrets/nginxfiles.json`.

This is a utility until kubectl gets something similar.
