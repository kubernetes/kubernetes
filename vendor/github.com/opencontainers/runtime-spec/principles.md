# <a name="the5PrinciplesOfStandardContainers" />The 5 principles of Standard Containers

Define a unit of software delivery called a Standard Container.
The goal of a Standard Container is to encapsulate a software component and all its dependencies in a format that is self-describing and portable, so that any compliant runtime can run it without extra dependencies, regardless of the underlying machine and the contents of the container.

The specification for Standard Containers defines:

1. configuration file formats
2. a set of standard operations
3. an execution environment.

A great analogy for this is the physical shipping container used by the transportation industry.
Shipping containers are a fundamental unit of delivery, they can be lifted, stacked, locked, loaded, unloaded and labelled.
Irrespective of their contents, by standardizing the container itself it allowed for a consistent, more streamlined and efficient set of processes to be defined.
For software Standard Containers offer similar functionality by being the fundamental, standardized, unit of delivery for a software package.

## <a name="standardOperations" />1. Standard operations

Standard Containers define a set of STANDARD OPERATIONS.
They can be created, started, and stopped using standard container tools; copied and snapshotted using standard filesystem tools; and downloaded and uploaded using standard network tools.

## <a name="contentAgnostic" />2. Content-agnostic

Standard Containers are CONTENT-AGNOSTIC: all standard operations have the same effect regardless of the contents.
They are started in the same way whether they contain a postgres database, a php application with its dependencies and application server, or Java build artifacts.

## <a name="infrastructureAgnostic" />3. Infrastructure-agnostic

Standard Containers are INFRASTRUCTURE-AGNOSTIC: they can be run in any OCI supported infrastructure.
For example, a standard container can be bundled on a laptop, uploaded to cloud storage, downloaded, run and snapshotted by a build server at a fiber hotel in Virginia, uploaded to 10 staging servers in a home-made private cloud cluster, then sent to 30 production instances across 3 public cloud regions.

## <a name="designedForAutomation" />4. Designed for automation

Standard Containers are DESIGNED FOR AUTOMATION: because they offer the same standard operations regardless of content and infrastructure, Standard Containers, are extremely well-suited for automation.
In fact, you could say automation is their secret weapon.

Many things that once required time-consuming and error-prone human effort can now be programmed.
Before Standard Containers, by the time a software component ran in production, it had been individually built, configured, bundled, documented, patched, vendored, templated, tweaked and instrumented by 10 different people on 10 different computers.
Builds failed, libraries conflicted, mirrors crashed, post-it notes were lost, logs were misplaced, cluster updates were half-broken.
The process was slow, inefficient and cost a fortune - and was entirely different depending on the language and infrastructure provider.

## <a name="industrialGradeDelivery" />5. Industrial-grade delivery

Standard Containers make INDUSTRIAL-GRADE DELIVERY of software a reality.
Leveraging all of the properties listed above, Standard Containers are enabling large and small enterprises to streamline and automate their software delivery pipelines.
Whether it is in-house devOps flows, or external customer-based software delivery mechanisms, Standard Containers are changing the way the community thinks about software packaging and delivery.
