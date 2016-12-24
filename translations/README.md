# Translations README

This is a basic sketch of the workflow needed to add translations:

# Adding/Updating Translations

## New languages
Create `translations/kubectl/<language>/LC_MESSAGES/k8s.po`. There's
no need to update `translations/test/...` which is only used for unit tests.

Move on to Adding new translations

## Adding new translations
Edit the appropriate `k8s.po` file, `poedit` is a popular open source tool
for translations.

Once you are done with your `.po` file, generate the corresponding `k8s.mo`
file. `poedit` does this automatically on save.

We use the English translation as both the `msg_id` and the `msg_context`.

## Regenerating the bindata file
Run `./hack/generate-bindata.sh, this will turn the translation files
into generated code which will in turn be packaged into the Kubernetes
binaries.

# Using translations

To use translations, you simply need to add:
```go
import pkg/i18n
...
// Get a translated string
translated := i18n.T("Your message in english here")

// Get a translated plural string
translated := i18n.T("You had % items", items)

// Translated error
return i18n.Error("Something bad happened")

// Translated plural error
return i18n.Error("%d bad things happened")
```


[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/translations/README.md?pixel)]()
