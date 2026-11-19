# PowerShell-based Static HTTP Server for Face Recognition Web App
# Run this script to start a local server at http://localhost:8000
# Press Ctrl+C in the terminal to stop the server

$port = 8000
$url = "http://localhost:$port/"
$listener = New-Object System.Net.HttpListener
$listener.Prefixes.Add($url)

try {
    $listener.Start()
    Write-Host ""
    Write-Host "==========================================================" -ForegroundColor Cyan
    Write-Host "  AI Face Recognition local web server is running!" -ForegroundColor Green
    Write-Host "  URL: $url" -ForegroundColor Yellow
    Write-Host "  Press Ctrl+C in this terminal window to stop the server." -ForegroundColor Red
    Write-Host "==========================================================" -ForegroundColor Cyan
    Write-Host ""
    
    # Try opening Google Chrome directly
    $chromePaths = @(
        "$env:ProgramFiles\Google\Chrome\Application\chrome.exe",
        "${env:ProgramFiles(x86)}\Google\Chrome\Application\chrome.exe",
        "$env:LOCALAPPDATA\Google\Chrome\Application\chrome.exe"
    )
    $launchedChrome = $false
    foreach ($p in $chromePaths) {
        if (Test-Path $p) {
            Start-Process $p -ArgumentList $url
            Write-Host "Launched Chrome from $p" -ForegroundColor Green
            $launchedChrome = $true
            break
        }
    }
    if (-not $launchedChrome) {
        Start-Process $url
    }

    while ($listener.IsListening) {
        try {
            $context = $listener.GetContext()
            $request = $context.Request
            $response = $context.Response
            
            $urlPath = $request.Url.LocalPath
            if ($urlPath -eq "/") {
                $urlPath = "/index.html"
            }
            
            # Replace forward slashes with backslashes for local path resolution
            $relPath = $urlPath.TrimStart("/").Replace("/", "\")
            $baseDir = if ($PSScriptRoot) { $PSScriptRoot } else { Get-Location }
            $filePath = Join-Path $baseDir $relPath
            
            if (Test-Path $filePath -PathType Leaf) {
                $extension = [System.IO.Path]::GetExtension($filePath).ToLower()
                $contentType = switch ($extension) {
                    ".html" { "text/html; charset=utf-8" }
                    ".css"  { "text/css; charset=utf-8" }
                    ".js"   { "application/javascript; charset=utf-8" }
                    ".json" { "application/json; charset=utf-8" }
                    ".png"  { "image/png" }
                    ".jpg"  { "image/jpeg" }
                    ".jpeg" { "image/jpeg" }
                    ".gif"  { "image/gif" }
                    ".svg"  { "image/svg+xml" }
                    ".ico"  { "image/x-icon" }
                    default { "application/octet-stream" }
                }
                
                $response.ContentType = $contentType
                $response.Headers.Add("Access-Control-Allow-Origin", "*")
                $response.Headers.Add("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
                $response.Headers.Add("Cache-Control", "no-cache, no-store, must-revalidate")
                
                $bytes = [System.IO.File]::ReadAllBytes($filePath)
                $response.ContentLength64 = $bytes.Length
                $response.OutputStream.Write($bytes, 0, $bytes.Length)
                Write-Host "[200] Serving $urlPath ($contentType)" -ForegroundColor Gray
            } else {
                $response.StatusCode = 404
                $response.ContentType = "text/plain"
                $buffer = [System.Text.Encoding]::UTF8.GetBytes("404 Not Found: $urlPath")
                $response.ContentLength64 = $buffer.Length
                $response.OutputStream.Write($buffer, 0, $buffer.Length)
                Write-Host "[404] Not Found: $urlPath" -ForegroundColor Red
            }
            $response.OutputStream.Close()
        }
        catch {
            # Catch per-request exceptions so the server loop never terminates unexpectedly
            Write-Host "[WARN] Request handling notice: $_" -ForegroundColor DarkGray
        }
    }
}
catch {
    Write-Host "Error starting server: $_" -ForegroundColor Red
}
finally {
    $listener.Stop()
    Write-Host "Server stopped." -ForegroundColor Yellow
}
