{{/*
Expand the name of the chart.
*/}}
{{- define "cloud-optimizer.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Create a default fully qualified app name.
*/}}
{{- define "cloud-optimizer.fullname" -}}
{{- if .Values.fullnameOverride }}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- $name := default .Chart.Name .Values.nameOverride }}
{{- if contains $name .Release.Name }}
{{- .Release.Name | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" }}
{{- end }}
{{- end }}
{{- end }}

{{/*
Common labels
*/}}
{{- define "cloud-optimizer.labels" -}}
helm.sh/chart: {{ include "cloud-optimizer.name" . }}-{{ .Chart.Version | replace "+" "_" }}
{{ include "cloud-optimizer.selectorLabels" . }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
app.kubernetes.io/part-of: cloud-optimizer
{{- end }}

{{/*
Selector labels
*/}}
{{- define "cloud-optimizer.selectorLabels" -}}
app.kubernetes.io/name: {{ include "cloud-optimizer.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}
