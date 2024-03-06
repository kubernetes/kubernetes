# v3.0.1

Fixed:
 - Security issue: an attacker specifying a large "p2c" value can cause
   JSONWebEncryption.Decrypt and JSONWebEncryption.DecryptMulti to consume large
   amounts of CPU, causing a DoS. Thanks to Matt Schwager (@mschwager) for the
   disclosure and to Tom Tervoort for originally publishing the category of attack.
   https://i.blackhat.com/BH-US-23/Presentations/US-23-Tervoort-Three-New-Attacks-Against-JSON-Web-Tokens.pdf
