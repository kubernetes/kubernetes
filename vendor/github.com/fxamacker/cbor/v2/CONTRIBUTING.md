# How to contribute

You can contribute by using the library, opening issues, or opening pull requests.

## Bug reports and security vulnerabilities

Most issues are tracked publicly on [GitHub](https://github.com/fxamacker/cbor/issues). 

To report security vulnerabilities, please email faye.github@gmail.com and allow time for the problem to be resolved before disclosing it to the public.  For more info, see [Security Policy](https://github.com/fxamacker/cbor#security-policy).

Please do not send data that might contain personally identifiable information, even if you think you have permission.  That type of support requires payment and a signed contract where I'm indemnified, held harmless, and defended by you for any data you send to me.

## Pull requests

Please [create an issue](https://github.com/fxamacker/cbor/issues/new/choose) before you begin work on a PR.  The improvement may have already been considered, etc.

Pull requests have signing requirements and must not be anonymous.  Exceptions are usually made for docs and CI scripts.

See the [Pull Request Template](https://github.com/fxamacker/cbor/blob/master/.github/pull_request_template.md) for details.

Pull requests have a greater chance of being approved if:
- it does not reduce speed, increase memory use, reduce security, etc. for people not using the new option or feature.
- it has > 97% code coverage.

## Describe your issue

Clearly describe the issue:
* If it's a bug, please provide: **version of this library** and **Go** (`go version`), **unmodified error message**, and describe **how to reproduce it**.  Also state **what you expected to happen** instead of the error.
* If you propose a change or addition, try to give an example how the improved code could look like or how to use it.
* If you found a compilation error, please confirm you're using a supported version of Go. If you are, then provide the output of `go version` first, followed by the complete error message.

## Please don't

Please don't send data containing personally identifiable information, even if you think you have permission.  That type of support requires payment and a contract where I'm indemnified, held harmless, and defended for any data you send to me.

Please don't send CBOR data larger than 1024 bytes by email. If you want to send crash-producing CBOR data > 1024 bytes by email, please get my permission before sending it to me.

## Credits

- This guide used nlohmann/json contribution guidelines for inspiration as suggested in issue #22.
- Special thanks to @lukseven for pointing out the contribution guidelines didn't mention signing requirements.
