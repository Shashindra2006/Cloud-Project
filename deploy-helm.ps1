# Automated Minikube & Helm Deployment Script
Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "  AI-Powered Cloud Optimizer - Kubernetes & Helm Deploy" -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan

# 1. Check Minikube
Write-Host "`n[1/5] Checking Minikube status..." -ForegroundColor Yellow
$status = minikube status --format "{{.Host}}" 2>$null
if ($status -ne "Running") {
    Write-Host "Starting Minikube cluster..." -ForegroundColor Cyan
    minikube start --driver=docker
} else {
    Write-Host "Minikube is already running." -ForegroundColor Green
}

# 2. Build Images inside Minikube
Write-Host "`n[2/5] Building Docker images inside Minikube..." -ForegroundColor Yellow
Write-Host "Building API image (cloudopt-api:latest)..." -ForegroundColor Cyan
minikube image build -t cloudopt-api:latest -f Dockerfile .

Write-Host "Building Dashboard image (cloudopt-dashboard:latest)..." -ForegroundColor Cyan
minikube image build -t cloudopt-dashboard:latest -f Dockerfile.dashboard .

# 3. Lint Helm Chart
Write-Host "`n[3/5] Validating Helm chart..." -ForegroundColor Yellow
helm lint ./helm/cloud-optimizer

# 4. Deploy with Helm
Write-Host "`n[4/5] Deploying chart via Helm..." -ForegroundColor Yellow
helm upgrade --install cloud-optimizer ./helm/cloud-optimizer `
    --set api.image.pullPolicy=Never `
    --set dashboard.image.pullPolicy=Never `
    --wait --timeout 5m

# 5. Display Status
Write-Host "`n[5/5] Deployment successful!" -ForegroundColor Green
Write-Host "`nPods:" -ForegroundColor Cyan
kubectl get pods -l app.kubernetes.io/instance=cloud-optimizer

Write-Host "`nServices:" -ForegroundColor Cyan
kubectl get svc -l app.kubernetes.io/instance=cloud-optimizer

Write-Host "`nHorizontal Pod Autoscaler (HPA):" -ForegroundColor Cyan
kubectl get hpa

Write-Host "`nTo open the Dashboard in your browser, run:" -ForegroundColor Yellow
Write-Host "minikube service cloud-optimizer-dashboard" -ForegroundColor Green
Write-Host "Or port-forward: kubectl port-forward svc/cloud-optimizer-dashboard 8501:8501" -ForegroundColor Green
