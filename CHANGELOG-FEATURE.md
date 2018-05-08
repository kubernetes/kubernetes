# Feature changelog: serverside apply

This is a list of things this feature branch changes from head. The point is not
history (you can get that from git) but keeping an up-to-date snapshot of the
changes. So, when you make a change it is legit to either add something here,
modify something, or even reshuffle the organization system.

Current organization is copied from the project map document.

    1. API Types and Toolchain (OpenAPI)
        1.1. New Go IDL Tags
        1.2. New OpenAPI extensions tags
        1.3. OpenAPI Compilation toolchain
        1.4. Fix-up existing APIs
        1.5. CRD (`validation` field) compatible spec
    2. Merging
        2.1. Ownership semantic
        2.2. Merge logic
    3. Machinery/Wiring
        3.1. New PATCH handler Content-Type
        3.2. Metadata API Change
            - Added a map[workflow-id]last-applied-state to metadata.
        3.3. OpenAPI data model wiring to strategy/registry
        3.4. Updated PUT Handler
        3.5. Admission chain
        3.6. Conversion/Defaulting
        3.7. Validation
    4. User Experience
        4.1. Kubectl
        4.2. Version skew and upgrade path
        4.3. Dry-run
    5. Feature Health
        5.1. Testing
        5.2. Monitoring
        5.3. Documentation
