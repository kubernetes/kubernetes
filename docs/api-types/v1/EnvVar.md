###EnvVar###

---
* name: 
  * **_type_**: string
  * **_description_**: name of the environment variable; must be a C_IDENTIFIER
* value: 
  * **_type_**: string
  * **_description_**: value of the environment variable; defaults to empty string; variable references $(VAR_NAME) are expanded using the previously defined environment varibles in the container and any service environment variables; if a variable cannot be resolved, the reference in the input string will be unchanged; the $(VAR_NAME) syntax can be escaped with a double $$, ie: $$(VAR_NAME) ; escaped references will never be expanded, regardless of whether the variable exists or not
* valueFrom: 
  * **_type_**: [EnvVarSource](EnvVarSource.md)
  * **_description_**: source for the environment variable's value; cannot be used if value is not empty
