[CmdletBinding(DefaultParameterSetName='Path')]
param (
    [Parameter(ParameterSetName='Path', Position=0, Mandatory=$true, ValueFromPipelineByPropertyName=$true)]
    [string[]] $Path,

    [Parameter(ParameterSetName='LiteralPath', Mandatory=$true, ValueFromPipelineByPropertyName=$true)]
    [Alias('PSPath')]
    [string[]] $LiteralPath,

    [Parameter()]
    [Alias('IncludeParents')]
    [switch] $AllowParentProducts,

    [Parameter()]
    [switch] $PassThru,

    [Parameter()]
    [switch] $Force
)

process {
    if ($PSCmdlet.ParameterSetName -eq 'Path') {
        $LiteralPath = Resolve-Path $Path
    }

    foreach ($p in $LiteralPath) {
        $file = Get-Item -LiteralPath $p
        if (!$Force -and @('.md', '.markdown') -notcontains $file.Extension) {
            Write-Verbose "Skipping $($file.FullName): does not appear to be a valid markdown file"
            continue
        }

        [string[]] $content = $file | Get-Content
        if (!$content -or !$content[0].StartsWith('---')) {
            Write-Verbose "Skipping $($file.FullName): does not contain frontmatter"
            continue
        }

        Write-Verbose "Checking $($file.FullName)"

        # Reset metadata and create mutable collections.
        $products = [System.Collections.Generic.List[string]]::new()

        $i = 1
        do {
            [string] $line = $content[$i++]
            if ($line.StartsWith('---')) {
                break
            }

            # For each interesting section, set $current to a [list[string]].
            if ($line -match '^\s*products:') {
                $current = $products
                $has_current = $true
            }
            elseif ($has_current -and $line -match '^\s*-\s+([\w-]+)') {
                $current.Add($matches[1])
            }
            elseif (![string]::IsNullOrWhiteSpace($line)) {
                $current = $null
                $has_current = $false
            }
        } while ($i -lt $content.Length)

        $has_errors = $false
        $invalidProducts = @()

        foreach ($product in $products) {
            if ($productSlugs -notcontains $product) {

                $has_errors = $true
                $invalidProducts += $product

                Write-Error "File '$($file.FullName)' contains invalid product slug: $product" -TargetObject $file `
                    -Category InvalidData -CategoryTargetName $product -CategoryTargetType string `
                    -RecommendedAction 'Use only product slugs listed at https://review.docs.microsoft.com/help/contribute/metadata-taxonomies?branch=master#product'
            }
        }

        if ($has_errors) {
            if ($PassThru) {
                $file | Add-Member -PassThru -Type NoteProperty -Name InvalidProducts -Value $invalidProducts `
                    | Add-Member -PassThru -Type PropertySet -Name SampleMetadata -Value @('InvalidProducts')
            }

            $script_has_errors = $true
        }
    }
}

end {
    if ($script_has_errors) {
        exit 1
    }
}

begin {
    # https://review.docs.microsoft.com/help/contribute/metadata-taxonomies?branch=master#product
    $productSlugs = @(
        "ai-builder",
        "aspnet",
        "aspnet-core",
        "azure-active-directory",
        "azure-active-directory-b2c",
        "azure-active-directory-domain",
        "azure-advisor",
        "azure-analysis-services",
        "azure-anomaly-detector",
        "azure-api-apps",
        "azure-api-fhir",
        "azure-api-management",
        "azure-app-configuration",
        "azure-app-service",
        "azure-app-service-mobile",
        "azure-app-service-static",
        "azure-app-service-web",
        "azure-application-gateway",
        "azure-application-insights",
        "azure-arc",
        "azure-archive-storage",
        "azure-artifacts",
        "azure-attestation",
        "azure-automation",
        "azure-avere-vFXT",
        "azure-backup",
        "azure-bastion",
        "azure-batch",
        "azure-bing-autosuggest",
        "azure-bing-custom",
        "azure-bing-entity",
        "azure-bing-image",
        "azure-bing-news",
        "azure-bing-spellcheck",
        "azure-bing-video",
        "azure-bing-visual",
        "azure-bing-web",
        "azure-blob-storage",
        "azure-blockchain-service",
        "azure-blockchain-tokens",
        "azure-blockchain-workbench",
        "azure-blueprints",
        "azure-boards",
        "azure-bot-service",
        "azure-cache-redis",
        "azure-cdn",
        "azure-clis",
        "azure-cloud-services",
        "azure-cloud-shell",
        "azure-cognitive-search",
        "azure-cognitive-services",
        "azure-communication-services",
        "azure-computer-vision",
        "azure-container-instances",
        "azure-container-registry",
        "azure-content-moderator",
        "azure-content-protection",
        "azure-cosmos-db",
        "azure-cost-management",
        "azure-custom-vision",
        "azure-cyclecloud",
        "azure-data-box-family",
        "azure-data-catalog",
        "azure-data-explorer",
        "azure-data-factory",
        "azure-data-lake",
        "azure-data-lake-analytics",
        "azure-data-lake-gen1",
        "azure-data-lake-gen2",
        "azure-data-lake-storage",
        "azure-data-science-vm",
        "azure-data-share",
        "azure-database-mariadb",
        "azure-database-migration",
        "azure-database-mysql",
        "azure-database-postgresql",
        "azure-databricks",
        "azure-ddos-protection",
        "azure-dedicated-host",
        "azure-dedicated-hsm",
        "azure-dev-spaces",
        "azure-dev-tool-integrations",
        "azure-devops",
        "azure-devops-tool-integrations",
        "azure-devtest-labs",
        "azure-digital-twins",
        "azure-disk-encryption",
        "azure-disk-storage",
        "azure-dns",
        "azure-encoding",
        "azure-event-grid",
        "azure-event-hubs",
        "azure-expressroute",
        "azure-face",
        "azure-farmbeats",
        "azure-files",
        "azure-firewall",
        "azure-firewall-manager",
        "azure-form-recognizer",
        "azure-front-door",
        "azure-functions",
        "azure-fxt-edge-filer",
        "azure-genomics",
        "azure-hdinsight",
        "azure-hdinsight-rserver",
        "azure-hpc-cache",
        "azure-immersive-reader",
        "azure-information-protection",
        "azure-ink-recognizer",
        "azure-internet-analyzer",
        "azure-iot",
        "azure-iot-central",
        "azure-iot-dps",
        "azure-iot-edge",
        "azure-iot-hub",
        "azure-iot-pnp",
        "azure-iot-sdk",
        "azure-iot-security-center",
        "azure-iot-solution-accelerators",
        "azure-key-vault",
        "azure-kinect-dk",
        "azure-kubernetes-service",
        "azure-lab-services",
        "azure-language-understanding",
        "azure-lighthouse",
        "azure-linux-vm",
        "azure-live-ondemand-streaming",
        "azure-live-video-analytics",
        "azure-load-balancer",
        "azure-log-analytics",
        "azure-logic-apps",
        "azure-machine-learning",
        "azure-machine-learning-designer",
        "azure-machine-learning-studio",
        "azure-managed-applications",
        "azure-managed-disks",
        "azure-maps",
        "azure-media-analytics",
        "azure-media-player",
        "azure-media-services",
        "azure-metrics-advisor",
        "azure-migrate",
        "azure-monitor",
        "azure-netapp-files",
        "azure-network-watcher",
        "azure-notebooks",
        "azure-notification-hubs",
        "azure-open-datasets",
        "azure-personalizer",
        "azure-pipelines",
        "azure-playfab",
        "azure-policy",
        "azure-portal",
        "azure-powerbi-embedded",
        "azure-private-link",
        "azure-qio",
        "azure-qna-maker",
        "azure-quantum",
        "azure-queue-storage",
        "azure-rbac",
        "azure-redhat-openshift",
        "azure-remote-rendering",
        "azure-repos",
        "azure-resource-graph",
        "azure-resource-manager",
        "azure-rtos",
        "azure-sap",
        "azure-scheduler",
        "azure-sdks",
        "azure-search",
        "azure-security-center",
        "azure-sentinel",
        "azure-service-bus",
        "azure-service-fabric",
        "azure-service-health",
        "azure-signalr-service",
        "azure-site-recovery",
        "azure-sovereign-china",
        "azure-sovereign-germany",
        "azure-sovereign-us",
        "azure-spatial-anchors",
        "azure-speaker-recognition",
        "azure-speech",
        "azure-speech-text",
        "azure-speech-translation",
        "azure-sphere",
        "azure-spring-cloud",
        "azure-sql-database",
        "azure-sql-edge",
        "azure-sql-managed-instance",
        "azure-sql-virtual-machines",
        "azure-sqlserver-stretchdb",
        "azure-sqlserver-vm",
        "azure-stack",
        "azure-stack-edge",
        "azure-stack-hci",
        "azure-stack-hub",
        "azure-storage",
        "azure-storage-accounts",
        "azure-storage-explorer",
        "azure-storsimple",
        "azure-stream-analytics",
        "azure-synapse-analytics",
        "azure-table-storage",
        "azure-test-plans",
        "azure-text-analytics",
        "azure-text-speech",
        "azure-time-series-insights",
        "azure-traffic-manager",
        "azure-translator",
        "azure-translator-speech",
        "azure-translator-text",
        "azure-video-indexer",
        "azure-virtual-machines",
        "azure-virtual-machines-windows",
        "azure-virtual-network",
        "azure-virtual-wan",
        "azure-vm-scalesets",
        "azure-vmware-solution",
        "azure-vpn-gateway",
        "azure-web-application-firewall",
        "azure-web-apps",
        "azure-webapp-containers",
        "blazor-server",
        "blazor-webassembly",
        "common-data-service",
        "customer-voice",
        "dotnet-core",
        "dotnet-standard",
        "dynamics-business-central",
        "dynamics-commerce",
        "dynamics-cust-insights",
        "dynamics-cust-svc-insights",
        "dynamics-customer-engagement",
        "dynamics-customer-service",
        "dynamics-field-service",
        "dynamics-finance",
        "dynamics-finance-operations",
        "dynamics-fraud-protection",
        "dynamics-guides",
        "dynamics-human-resources",
        "dynamics-layout",
        "dynamics-market-insights",
        "dynamics-marketing",
        "dynamics-prod-visualize",
        "dynamics-product-insights",
        "dynamics-project-operations",
        "dynamics-project-service",
        "dynamics-remote-assist",
        "dynamics-retail",
        "dynamics-sales",
        "dynamics-sales-insights",
        "dynamics-scm",
        "dynamics-talent",
        "dynamics-talent-attract",
        "dynamics-talent-core",
        "dynamics-talent-onboard",
        "ef-core",
        "ef6",
        "expression-studio",
        "m365-ems",
        "m365-ems-cloud-app-security",
        "m365-ems-configuration-manager",
        "m365-information-protection",
        "m365-myanalytics",
        "m365-security-center",
        "m365-security-score",
        "m365-threat-protection",
        "m365-workplace-analytics",
        "mem-configuration-manager",
        "mem-intune",
        "microsoft-identity-web",
        "mlnet",
        "msal-android",
        "msal-angular",
        "msal-ios",
        "msal-java",
        "msal-js",
        "msal-node",
        "msal-python",
        "msc-operations-manager",
        "msc-service-manager",
        "mscloud-financial",
        "mscloud-healthcare",
        "mscloud-manufacturing",
        "mscloud-nonprofit",
        "mscloud-retail",
        "office-365-atp",
        "office-access",
        "office-adaptive-cards",
        "office-add-ins",
        "office-bookings",
        "office-excel",
        "office-exchange-server",
        "office-forefront",
        "office-kaizala",
        "office-lync-server",
        "office-onedrive",
        "office-onenote",
        "office-outlook",
        "office-planner",
        "office-powerpoint",
        "office-project",
        "office-project-server",
        "office-publisher",
        "office-skype-business",
        "office-sp",
        "office-sp-designer",
        "office-sp-framework",
        "office-sp-server",
        "office-ui-fabric",
        "office-visio",
        "office-word",
        "office-yammer",
        "passport-azure-ad",
        "power-apps",
        "power-automate",
        "power-bi",
        "power-query",
        "power-virtual-agents",
        "return-to-school",
        "return-to-workplace",
        "sql-server-2008",
        "surface-duo",
        "sway",
        "vs-app-center",
        "vs-code",
        "vs-mac",
        "vs-online",
        "windows-api-win32",
        "windows-azure-pack",
        "windows-forms",
        "windows-iot",
        "windows-iot-10core",
        "windows-mdop",
        "windows-mixed-reality",
        "windows-server",
        "windows-smb-server",
        "windows-system-center",
        "windows-uwp",
        "windows-virtual-desktop",
        "windows-wdk",
        "windows-wpf",
        "xamarin"
    )

    if ($AllowParentProducts) {
        $productSlugs += @(
            "azure",
            "bing",
            "blazor",
            "connected-services-framework",
            "consumer",
            "customer-care-framework",
            "dotnet",
            "dynamics",
            "dynamics-365",
            "expression",
            "flipgrid",
            "github",
            "hololens",
            "industry-solutions",
            "internet-explorer",
            "kinect",
            "m365",
            "makecode",
            "mdatp",
            "mem",
            "microsoft-authentication-library",
            "microsoft-edge",
            "microsoft-mesh",
            "microsoft-servers",
            "minecraft",
            "mrtk",
            "ms-graph",
            "msc",
            "office",
            "office-365",
            "office-teams",
            "power-platform",
            "project-acoustics",
            "qdk",
            "silverlight",
            "skype",
            "sql-server",
            "surface",
            "vs",
            "windows",
            "xbox"
        )
    }
}

<#
.SYNOPSIS
Checks sample markdown files' frontmatter for invalid information.

.DESCRIPTION
Given a collection of markdown files, their frontmatter - if present - is checked for invalid information, including:

Invalid product slugs, i.e. those not listed in https://review.docs.microsoft.com/help/contribute/metadata-taxonomies?branch=master#product.

.PARAMETER Path
Specifies the path to an item to search. Wildcards are permitted.

.PARAMETER LiteralPath
Specifies the path to an item to search. Wildcards are not permitted.

.PARAMETER AllowParentProducts
Allow parent product slugs, like "azure" for "azure-key-vault".

.PARAMETER PassThru
By default, any invalid information is written to the $Error stream. Pass -PassThru to also return file items with error information attached.

.PARAMETER Force
Ignore file type validation.

.EXAMPLE
Get-ChildItem sdk -Filter *.md -Recurse | Test-SampleMetadata.ps1 -AllowParentProducts

Searches all markdown (*.md) files under an "sdk" subdirectory for invalid frontmatter.

.EXAMPLE
Test-SampleMetadata.ps1 sample\README.md -PassThru | Select-Object FullName, SampleMetadata

Shows sample metadata parsed and attached to the specified file object.

.EXAMPLE
Get-ChildItem sdk -Filter *.sample -Recurse | Test-SampleMetadata.ps1 -Force

Searches for all .sample files and ignores file type validation within the script, which may lead to extraneous errors.
#>
