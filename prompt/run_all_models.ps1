$models = @(
    "gpt-oss-120b",
    "mistral-large-3-675b-instruct-2512",
    "qwen3.5-122b-a10b"
)

foreach ($model in $models) {
    Write-Host "========================================"
    Write-Host "Running with model: $model"
    Write-Host "========================================"
    $env:MODEL_NAME = $model
    python prompt\process_long_transcripts_with_profiles.py
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[ERROR] Model $model failed with exit code $LASTEXITCODE" -ForegroundColor Red
    }
    Write-Host ""
}

Write-Host "========================================"
Write-Host "All models complete."
Write-Host "========================================"
