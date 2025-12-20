$experiments = @(
    "sst2_mixture_soft_falcon"
    "ag_news_mixture_soft_falcon"
    "trec_mixture_soft_falcon"
)

foreach ($exp in $experiments) {
    Write-Host "`n=== Running experiment: $exp ===" -ForegroundColor Cyan
    py -3.11 -m src.run_experiment --config-name=$exp
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Experiment $exp FAILED (exit code $LASTEXITCODE), continuing…" -ForegroundColor Yellow
    }
}

Write-Host "`nAll done!" -ForegroundColor Green