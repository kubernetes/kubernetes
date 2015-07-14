###ExecAction###

---
* command: 
  * **_type_**: []string
  * **_description_**: command line to execute inside the container; working directory for the command is root ('/') in the container's file system; the command is exec'd, not run inside a shell; exit status of 0 is treated as live/healthy and non-zero is unhealthy
