{% panel style="success", title="Providing Feedback" %}
**Provide feedback at the [survey](https://www.surveymonkey.com/r/JH35X82)**
{% endpanel %}

# Introduction

While Declarative Management of Applications is the recommended pattern for production
use cases, imperative porcelain commands may be helpful for development or debugging
issues.  These commands are particularly helpful for learning about Kubernetes when coming
from an imperative system.

**Note:** Some imperative commands can be run with `--dry-run -o yaml` to display the declarative
form.

This section describes imperative commands that will generate or patch Resource Config.
