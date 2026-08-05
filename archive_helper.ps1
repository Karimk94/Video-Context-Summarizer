<#
.SYNOPSIS
    Archive helper for create_archive.bat
    Packages all project source files + offline wheels into a ZIP.

.PARAMETER ZipDest
    Full path to the destination .zip file to create.
#>
param(
    [Parameter(Mandatory = $true)]
    [string]$ZipDest
)

Set-StrictMode -Off
$ErrorActionPreference = 'Stop'

$src      = $PSScriptRoot
$excludes = @('.git', '.gitignore', 'create_archive.bat', 'venv', '__pycache__', 'graphify-out', 'logs', 'uploads')

function Test-Excluded($relPath) {
    foreach ($p in $excludes) {
        if (($relPath -eq $p) -or ($relPath.StartsWith("$p/")) -or ($relPath.StartsWith("$p\"))) {
            return $true
        }
    }
    return $false
}

# -- Collect files via git (tracked + untracked non-ignored) --
Write-Host "Scanning files with git..."
$files = @(git ls-files -c --others --exclude-standard | Where-Object { -not (Test-Excluded $_) })

Write-Host "Including $($files.Count) source files..."
if ($files.Count -eq 0) {
    Write-Warning "No files found - nothing to archive."
    exit 0
}

# -- Staging directory --
$tempDir = Join-Path $env:TEMP ([System.IO.Path]::GetRandomFileName())
New-Item -ItemType Directory -Path $tempDir -Force | Out-Null

try {
    # -- Copy source files --
    foreach ($rel in $files) {
        $sourcePath = Join-Path $src $rel
        if (-not (Test-Path $sourcePath -PathType Leaf)) {
            Write-Warning "Skipped missing file: $rel"
            continue
        }
        $target = Join-Path $tempDir ($rel -replace '/', '\')
        $parent = Split-Path $target -Parent
        if (-not (Test-Path $parent)) {
            New-Item -ItemType Directory -Path $parent -Force | Out-Null
        }
        Copy-Item -Path $sourcePath -Destination $target
    }

    # -- Copy offline packages folder --
    $pkgSrc = Join-Path $src 'packages'
    if (Test-Path $pkgSrc) {
        $pkgCount = (Get-ChildItem $pkgSrc -File -Recurse).Count
        Write-Host "Including $pkgCount offline wheel files from packages..."
        $pkgDest = Join-Path $tempDir 'packages'
        New-Item -ItemType Directory -Path $pkgDest -Force | Out-Null
        Copy-Item -Path (Join-Path $pkgSrc '*') -Destination $pkgDest -Recurse -Force
    } else {
        Write-Warning "packages folder not found - archive will NOT be self-contained."
        Write-Warning "Run download_packages.bat on an internet-connected machine first."
    }

    # -- Compress --
    Write-Host "Compressing archive..."
    Compress-Archive -Path (Join-Path $tempDir '*') -DestinationPath $ZipDest -CompressionLevel Optimal
    Write-Host "Created: $ZipDest"

} finally {
    # Always clean up staging dir
    Remove-Item -Path $tempDir -Recurse -Force -ErrorAction SilentlyContinue
}
