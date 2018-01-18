# Node e2e job migration notice:

Sig-testing is actively migrating node e2e jobs from Jenkins to [Prow],
and we are moving *.property and image-config.yaml files to [test-infra]

If you want to update those files, please also update them in [test-infra].

If you have any questions, please contact @krzyzacy or #sig-testing.


## Test-infra Links:
Here's where the existing node e2e job config live:

[Image config files](https://github.com/kubernetes/test-infra/tree/master/jobs/e2e_node)

[Node test job args (.properties equivalent)](https://github.com/kubernetes/test-infra/blob/master/jobs/config.json)


[test-infra]: https://github.com/kubernetes/test-infra
[Prow]: https://github.com/kubernetes/test-infra/tree/master/prow
