Set up test keys.

  $ cat > rsa.pub <<EOF
  > -----BEGIN PUBLIC KEY-----
  > MIIBIjANBgkqhkiG9w0BAQEFAAOCAQ8AMIIBCgKCAQEAslWybuiNYR7uOgKuvaBw
  > qVk8saEutKhOAaW+3hWF65gJei+ZV8QFfYDxs9ZaRZlWAUMtncQPnw7ZQlXO9ogN
  > 5cMcN50C6qMOOZzghK7danalhF5lUETC4Hk3Eisbi/PR3IfVyXaRmqL6X66MKj/J
  > AKyD9NFIDVy52K8A198Jojnrw2+XXQW72U68fZtvlyl/BTBWQ9Re5JSTpEcVmpCR
  > 8FrFc0RPMBm+G5dRs08vvhZNiTT2JACO5V+J5ZrgP3s5hnGFcQFZgDnXLInDUdoi
  > 1MuCjaAU0ta8/08pHMijNix5kFofdPEB954MiZ9k4kQ5/utt02I9x2ssHqw71ojj
  > vwIDAQAB
  > -----END PUBLIC KEY-----
  > EOF

  $ cat > rsa.key <<EOF
  > -----BEGIN RSA PRIVATE KEY-----
  > MIIEogIBAAKCAQEAslWybuiNYR7uOgKuvaBwqVk8saEutKhOAaW+3hWF65gJei+Z
  > V8QFfYDxs9ZaRZlWAUMtncQPnw7ZQlXO9ogN5cMcN50C6qMOOZzghK7danalhF5l
  > UETC4Hk3Eisbi/PR3IfVyXaRmqL6X66MKj/JAKyD9NFIDVy52K8A198Jojnrw2+X
  > XQW72U68fZtvlyl/BTBWQ9Re5JSTpEcVmpCR8FrFc0RPMBm+G5dRs08vvhZNiTT2
  > JACO5V+J5ZrgP3s5hnGFcQFZgDnXLInDUdoi1MuCjaAU0ta8/08pHMijNix5kFof
  > dPEB954MiZ9k4kQ5/utt02I9x2ssHqw71ojjvwIDAQABAoIBABrYDYDmXom1BzUS
  > PE1s/ihvt1QhqA8nmn5i/aUeZkc9XofW7GUqq4zlwPxKEtKRL0IHY7Fw1s0hhhCX
  > LA0uE7F3OiMg7lR1cOm5NI6kZ83jyCxxrRx1DUSO2nxQotfhPsDMbaDiyS4WxEts
  > 0cp2SYJhdYd/jTH9uDfmt+DGwQN7Jixio1Dj3vwB7krDY+mdre4SFY7Gbk9VxkDg
  > LgCLMoq52m+wYufP8CTgpKFpMb2/yJrbLhuJxYZrJ3qd/oYo/91k6v7xlBKEOkwD
  > 2veGk9Dqi8YPNxaRktTEjnZb6ybhezat93+VVxq4Oem3wMwou1SfXrSUKtgM/p2H
  > vfw/76ECgYEA2fNL9tC8u9M0wjA+kvvtDG96qO6O66Hksssy6RWInD+Iqk3MtHQt
  > LeoCjvX+zERqwOb6SI6empk5pZ9E3/9vJ0dBqkxx3nqn4M/nRWnExGgngJsL959t
  > f50cdxva8y1RjNhT4kCwTrupX/TP8lAG8SfG1Alo2VFR8iWd8hDQcTECgYEA0Xfj
  > EgqAsVh4U0s3lFxKjOepEyp0G1Imty5J16SvcOEAD1Mrmz94aSSp0bYhXNVdbf7n
  > Rk77htWC7SE29fGjOzZRS76wxj/SJHF+rktHB2Zt23k1jBeZ4uLMPMnGLY/BJ099
  > 5DTGo0yU0rrPbyXosx+ukfQLAHFuggX4RNeM5+8CgYB7M1J/hGMLcUpjcs4MXCgV
  > XXbiw2c6v1r9zmtK4odEe42PZ0cNwpY/XAZyNZAAe7Q0stxL44K4NWEmxC80x7lX
  > ZKozz96WOpNnO16qGC3IMHAT/JD5Or+04WTT14Ue7UEp8qcIQDTpbJ9DxKk/eglS
  > jH+SIHeKULOXw7fSu7p4IQKBgBnyVchIUMSnBtCagpn4DKwDjif3nEY+GNmb/D2g
  > ArNiy5UaYk5qwEmV5ws5GkzbiSU07AUDh5ieHgetk5dHhUayZcOSLWeBRFCLVnvU
  > i0nZYEZNb1qZGdDG8zGcdNXz9qMd76Qy/WAA/nZT+Zn1AiweAovFxQ8a/etRPf2Z
  > DbU1AoGAHpCgP7B/4GTBe49H0AQueQHBn4RIkgqMy9xiMeR+U+U0vaY0TlfLhnX+
  > 5PkNfkPXohXlfL7pxwZNYa6FZhCAubzvhKCdUASivkoGaIEk6g1VTVYS/eDVQ4CA
  > slfl+elXtLq/l1kQ8C14jlHrQzSXx4PQvjDEnAmaHSJNz4mP9Fg=
  > -----END RSA PRIVATE KEY-----
  > EOF

  $ cat > ec.pub <<EOF
  > -----BEGIN PUBLIC KEY-----
  > MHYwEAYHKoZIzj0CAQYFK4EEACIDYgAE9yoUEAgxTd9svwe9oPqjhcP+f2jcdTL2
  > Wq8Aw2v9ht1dBy00tFRPNrCxFCkvMcJFhSPoDUV5NL7zfh3/psiSNYziGPrWEJYf
  > gmYihjSeoOf0ru1erpBrTflImPrMftCy
  > -----END PUBLIC KEY-----
  > EOF

  $ cat > ec.key <<EOF
  > -----BEGIN EC PRIVATE KEY-----
  > MIGkAgEBBDDvoj/bM1HokUjYWO/IDFs26Jo0GIFtU3tMQQu7ZabKscDMK3dZA0mK
  > v97ij7BBFbCgBwYFK4EEACKhZANiAAT3KhQQCDFN32y/B72g+qOFw/5/aNx1MvZa
  > rwDDa/2G3V0HLTS0VE82sLEUKS8xwkWFI+gNRXk0vvN+Hf+myJI1jOIY+tYQlh+C
  > ZiKGNJ6g5/Su7V6ukGtN+UiY+sx+0LI=
  > -----END EC PRIVATE KEY-----
  > EOF

Encrypt and then decrypt a test message (RSA).

  $ echo "Lorem ipsum dolor sit amet" | 
  > jose-util encrypt --alg RSA-OAEP --enc A128GCM --key rsa.pub |
  > jose-util decrypt --key rsa.key
  Lorem ipsum dolor sit amet

Encrypt and then decrypt a test message (EC).

  $ echo "Lorem ipsum dolor sit amet" | 
  > jose-util encrypt --alg ECDH-ES+A128KW --enc A128GCM --key ec.pub |
  > jose-util decrypt --key ec.key
  Lorem ipsum dolor sit amet

Sign and verify a test message (RSA).

  $ echo "Lorem ipsum dolor sit amet" | 
  > jose-util sign --alg PS256 --key rsa.key |
  > jose-util verify --key rsa.pub
  Lorem ipsum dolor sit amet

Sign and verify a test message (EC).

  $ echo "Lorem ipsum dolor sit amet" | 
  > jose-util sign --alg ES384 --key ec.key |
  > jose-util verify --key ec.pub
  Lorem ipsum dolor sit amet

Expand a compact message to full format.

  $ echo "eyJhbGciOiJFUzM4NCJ9.TG9yZW0gaXBzdW0gZG9sb3Igc2l0IGFtZXQK.QPU35XY913Im7ZEaN2yHykfbtPqjHZvYp-lV8OcTAJZs67bJFSdTSkQhQWE9ch6tvYrj_7py6HKaWVFLll_s_Rm6bmwq3JszsHrIvFFm1NydruYHhvAnx7rjYiqwOu0W" |
  > jose-util expand --format JWS
  {"payload":"TG9yZW0gaXBzdW0gZG9sb3Igc2l0IGFtZXQK","protected":"eyJhbGciOiJFUzM4NCJ9","signature":"QPU35XY913Im7ZEaN2yHykfbtPqjHZvYp-lV8OcTAJZs67bJFSdTSkQhQWE9ch6tvYrj_7py6HKaWVFLll_s_Rm6bmwq3JszsHrIvFFm1NydruYHhvAnx7rjYiqwOu0W"}
