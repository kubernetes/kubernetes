BigQuery Storage Driver
=======

[EXPERIMENTAL] Support for BigQuery backend as cAdvisor storage driver.
The current implementation takes bunch of BigQuery specific flags for authentication.
These will be merged into a single backend config.

To run the current version, following flags need to be specified:
```
 # Storage driver to use.
 -storage_driver=bigquery
 
 # Information about server-to-server Oauth token.
 # These can be obtained by creating a Service Account client id under `Google Developer API`
 
 # service client id
 -bq_id="XYZ.apps.googleusercontent.com"
 
 # service email address
 -bq_account="ABC@developer.gserviceaccount.com"
 
 # path to pem key (converted from p12 file)
 -bq_credentials_file="/path/to/key.pem"
 
 # project id to use for storing datasets.
 -bq_project_id="awesome_project"
```

See [Service account Authentication](https://developers.google.com/accounts/docs/OAuth2) for Oauth related details.
