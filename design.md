# LiveDirector AI - Design Document

## 1. System Architecture Overview

LiveDirector AI follows an event-driven serverless architecture optimized for real-time processing and scalability.

### 1.1 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        User Browser                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │ React UI     │  │ WebRTC       │  │ Canvas       │          │
│  │ Components   │  │ Audio Stream │  │ Compositor   │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
└─────────────────────────────────────────────────────────────────┘
                            │
                            │ WebSocket (WSS)
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                      AWS API Gateway                             │
│                    (WebSocket API)                               │
└─────────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                    AWS Lambda Functions                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │ Connection   │  │ Message      │  │ Disconnect   │          │
│  │ Handler      │  │ Handler      │  │ Handler      │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
└─────────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Core Processing Pipeline                      │
└─────────────────────────────────────────────────────────────────┘
```

### 1.2 Core Components

1. **Frontend Layer**: Browser-based UI with real-time video composition
2. **API Gateway Layer**: WebSocket management and routing
3. **Processing Layer**: Serverless functions for audio/video processing
4. **AI/ML Layer**: Bedrock models for reasoning and embeddings
5. **Storage Layer**: S3 for assets, DynamoDB for state
6. **Search Layer**: Vector database for semantic retrieval

## 2. Detailed Component Design

### 2.1 Frontend Architecture

#### 2.1.1 Technology Stack
- **Framework**: React 18+ with TypeScript
- **Styling**: Tailwind CSS 3.x
- **State Management**: Zustand (lightweight alternative to Redux)
- **Video Processing**: HTML5 Canvas API + MediaRecorder API
- **WebSocket Client**: native WebSocket API with reconnection logic

#### 2.1.2 Component Hierarchy

```
App
├── AuthProvider
│   └── LoginPage / SignupPage
├── DashboardLayout
│   ├── Header (user info, quota display)
│   ├── Sidebar (navigation, settings)
│   └── MainContent
│       ├── StudioPage
│       │   ├── VideoPreview (Canvas-based)
│       │   ├── RecordingControls
│       │   ├── TranscriptionPanel
│       │   └── BRollSuggestions
│       ├── LibraryPage
│       │   ├── AssetGrid
│       │   └── UploadModal
│       └── ExportPage
│           ├── ExportSettings
│           └── DownloadManager
└── NotificationProvider
```


#### 2.1.3 Video Composition Engine

The Canvas-based compositor handles real-time video overlay:

```typescript
class VideoCompositor {
  private canvas: HTMLCanvasElement;
  private ctx: CanvasRenderingContext2D;
  private webcamStream: MediaStream;
  private brollVideo: HTMLVideoElement;
  
  // Layout modes
  private layouts = {
    fullscreen: { webcam: null, broll: [0, 0, 1920, 1080] },
    pip: { webcam: [1520, 880, 400, 200], broll: [0, 0, 1920, 1080] },
    sideBySide: { webcam: [0, 0, 960, 1080], broll: [960, 0, 960, 1080] }
  };
  
  render() {
    // Composite frame at 60fps
    requestAnimationFrame(() => this.render());
    
    // Draw B-roll layer
    if (this.brollVideo.readyState === 4) {
      this.ctx.drawImage(this.brollVideo, ...this.layouts.pip.broll);
    }
    
    // Draw webcam layer
    this.ctx.drawImage(this.webcamStream, ...this.layouts.pip.webcam);
  }
  
  switchBRoll(videoUrl: string) {
    // Preload and crossfade to new B-roll
    const newVideo = document.createElement('video');
    newVideo.src = videoUrl;
    newVideo.oncanplay = () => {
      this.brollVideo = newVideo;
    };
  }
}
```

#### 2.1.4 WebSocket Communication

```typescript
class LiveDirectorWebSocket {
  private ws: WebSocket;
  private reconnectAttempts = 0;
  private maxReconnectAttempts = 5;
  
  connect(sessionId: string) {
    this.ws = new WebSocket(
      `wss://api.livedirector.ai/ws?sessionId=${sessionId}`
    );
    
    this.ws.onmessage = (event) => {
      const message = JSON.parse(event.data);
      this.handleMessage(message);
    };
    
    this.ws.onerror = () => this.reconnect();
  }
  
  sendAudioChunk(audioData: ArrayBuffer) {
    if (this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(audioData);
    }
  }
  
  handleMessage(message: WSMessage) {
    switch (message.type) {
      case 'transcription':
        store.updateTranscript(message.text);
        break;
      case 'broll_suggestion':
        store.addBRollSuggestion(message.clips);
        break;
      case 'overlay_command':
        videoCompositor.switchBRoll(message.videoUrl);
        break;
    }
  }
}
```

### 2.2 Backend Architecture

#### 2.2.1 API Gateway Configuration

```yaml
WebSocketAPI:
  Type: AWS::ApiGatewayV2::Api
  Properties:
    Name: LiveDirectorWebSocket
    ProtocolType: WEBSOCKET
    RouteSelectionExpression: "$request.body.action"
    
Routes:
  - $connect: ConnectFunction
  - $disconnect: DisconnectFunction
  - $default: MessageHandlerFunction
  - sendAudio: AudioProcessorFunction
```

#### 2.2.2 Lambda Function Architecture

**Connection Handler**
```python
# lambda/connect_handler.py
import boto3
import json

dynamodb = boto3.resource('dynamodb')
table = dynamodb.Table('LiveDirectorSessions')

def handler(event, context):
    connection_id = event['requestContext']['connectionId']
    session_id = event['queryStringParameters']['sessionId']
    
    # Store connection mapping
    table.put_item(Item={
        'sessionId': session_id,
        'connectionId': connection_id,
        'timestamp': int(time.time()),
        'status': 'connected'
    })
    
    return {'statusCode': 200, 'body': 'Connected'}
```

**Message Handler (Core Pipeline)**
```python
# lambda/message_handler.py
import json
import asyncio
from services.transcription import TranscriptionService
from services.reasoning import ReasoningService
from services.retrieval import RetrievalService

transcription_svc = TranscriptionService()
reasoning_svc = ReasoningService()
retrieval_svc = RetrievalService()

async def handler(event, context):
    connection_id = event['requestContext']['connectionId']
    body = json.loads(event['body'])
    
    if body['action'] == 'sendAudio':
        # Step 1: Transcribe audio chunk
        audio_data = base64.b64decode(body['audioData'])
        transcript = await transcription_svc.transcribe(audio_data)
        
        # Send partial transcript back to client
        await send_to_client(connection_id, {
            'type': 'transcription',
            'text': transcript['text'],
            'isFinal': transcript['isFinal']
        })
        
        # Step 2: Analyze context (only for final transcripts)
        if transcript['isFinal']:
            context = await reasoning_svc.analyze(transcript['text'])
            
            # Step 3: Retrieve relevant B-roll
            clips = await retrieval_svc.search(context['query'])
            
            # Step 4: Send overlay command
            await send_to_client(connection_id, {
                'type': 'overlay_command',
                'videoUrl': clips[0]['url'],
                'confidence': clips[0]['score']
            })
    
    return {'statusCode': 200}
```


### 2.3 AI/ML Services Design

#### 2.3.1 Transcription Service

```python
# services/transcription.py
import boto3
from amazon_transcribe.client import TranscribeStreamingClient
from amazon_transcribe.handlers import TranscriptResultStreamHandler

class TranscriptionService:
    def __init__(self):
        self.client = TranscribeStreamingClient(region="ap-south-1")
        self.language_code = "hi-IN"  # Auto-detect in production
    
    async def transcribe(self, audio_stream):
        """
        Streaming transcription with partial results
        """
        stream = await self.client.start_stream_transcription(
            language_code=self.language_code,
            media_sample_rate_hz=16000,
            media_encoding="pcm"
        )
        
        async for event in stream.output_stream:
            for result in event.transcript.results:
                if result.alternatives:
                    transcript = result.alternatives[0].transcript
                    return {
                        'text': transcript,
                        'isFinal': not result.is_partial,
                        'confidence': result.alternatives[0].confidence
                    }
```

#### 2.3.2 Reasoning Service (Amazon Bedrock)

```python
# services/reasoning.py
import boto3
import json

class ReasoningService:
    def __init__(self):
        self.bedrock = boto3.client('bedrock-runtime', region_name='us-east-1')
        self.model_id = "anthropic.claude-3-haiku-20240307-v1:0"
    
    async def analyze(self, transcript_text: str) -> dict:
        """
        Analyze transcript to extract context and generate search query
        """
        prompt = f"""Analyze this spoken text and extract:
1. Main topic/subject
2. Sentiment (positive/negative/neutral)
3. Visual concepts that would make good B-roll footage
4. A search query for finding relevant video clips

Text: "{transcript_text}"

Respond in JSON format:
{{
  "topic": "...",
  "sentiment": "...",
  "visual_concepts": ["...", "..."],
  "query": "..."
}}"""
        
        body = json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 500,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": 0.3
        })
        
        response = self.bedrock.invoke_model(
            modelId=self.model_id,
            body=body
        )
        
        result = json.loads(response['body'].read())
        context = json.loads(result['content'][0]['text'])
        
        return context
```

#### 2.3.3 Retrieval Service (Vector Search)

```python
# services/retrieval.py
import boto3
from opensearchpy import OpenSearch, RequestsHttpConnection
from requests_aws4auth import AWS4Auth

class RetrievalService:
    def __init__(self):
        self.bedrock = boto3.client('bedrock-runtime')
        self.opensearch = self._init_opensearch()
        self.index_name = "video-assets"
    
    def _init_opensearch(self):
        credentials = boto3.Session().get_credentials()
        awsauth = AWS4Auth(
            credentials.access_key,
            credentials.secret_key,
            'ap-south-1',
            'es',
            session_token=credentials.token
        )
        
        return OpenSearch(
            hosts=[{'host': 'search-livedirector.ap-south-1.es.amazonaws.com', 'port': 443}],
            http_auth=awsauth,
            use_ssl=True,
            connection_class=RequestsHttpConnection
        )
    
    async def search(self, context: dict, top_k: int = 3) -> list:
        """
        Semantic search using vector embeddings
        """
        # Generate query embedding
        query_embedding = await self._generate_embedding(context['query'])
        
        # KNN search in OpenSearch
        search_body = {
            "size": top_k,
            "query": {
                "knn": {
                    "embedding": {
                        "vector": query_embedding,
                        "k": top_k
                    }
                }
            },
            "_source": ["url", "title", "tags", "duration"]
        }
        
        response = self.opensearch.search(
            index=self.index_name,
            body=search_body
        )
        
        # Format results
        clips = []
        for hit in response['hits']['hits']:
            clips.append({
                'url': hit['_source']['url'],
                'title': hit['_source']['title'],
                'score': hit['_score'],
                'duration': hit['_source']['duration']
            })
        
        return clips
    
    async def _generate_embedding(self, text: str) -> list:
        """
        Generate embedding using Amazon Titan
        """
        body = json.dumps({
            "inputText": text
        })
        
        response = self.bedrock.invoke_model(
            modelId="amazon.titan-embed-text-v1",
            body=body
        )
        
        result = json.loads(response['body'].read())
        return result['embedding']
```

### 2.4 Data Models

#### 2.4.1 DynamoDB Tables

**Sessions Table**
```
Table: LiveDirectorSessions
Partition Key: sessionId (String)
Sort Key: timestamp (Number)

Attributes:
- connectionId: String
- userId: String
- status: String (connected|processing|completed)
- transcriptBuffer: String
- currentBRoll: String
- metadata: Map
```

**Users Table**
```
Table: LiveDirectorUsers
Partition Key: userId (String)

Attributes:
- email: String
- tier: String (free|pro)
- usageMinutes: Number
- quotaLimit: Number
- createdAt: Number
- lastLogin: Number
```

**Assets Table**
```
Table: VideoAssets
Partition Key: assetId (String)

Attributes:
- url: String (S3 URL)
- title: String
- description: String
- tags: List<String>
- category: String
- duration: Number
- uploadedBy: String
- isPublic: Boolean
```

#### 2.4.2 OpenSearch Index Schema

```json
{
  "mappings": {
    "properties": {
      "assetId": { "type": "keyword" },
      "title": { "type": "text" },
      "description": { "type": "text" },
      "tags": { "type": "keyword" },
      "category": { "type": "keyword" },
      "embedding": {
        "type": "knn_vector",
        "dimension": 1536,
        "method": {
          "name": "hnsw",
          "space_type": "cosinesimil",
          "engine": "nmslib"
        }
      },
      "url": { "type": "keyword" },
      "duration": { "type": "integer" },
      "sentiment": { "type": "keyword" }
    }
  }
}
```


### 2.5 Processing Pipeline Flow

#### 2.5.1 Real-Time Processing Sequence

```
User Speaks
    │
    ▼
[1] Audio Capture (Browser)
    │ MediaRecorder API captures 1-second chunks
    │ Format: PCM 16kHz mono
    ▼
[2] WebSocket Send
    │ Base64 encoded audio chunk
    │ Latency target: < 100ms
    ▼
[3] Lambda: Audio Processor
    │ Decode audio
    │ Buffer management (sliding window)
    ▼
[4] Amazon Transcribe Streaming
    │ Real-time speech-to-text
    │ Partial results every 500ms
    │ Final results on sentence boundary
    │ Latency: ~800ms
    ▼
[5] Lambda: Reasoning Engine
    │ Triggered only on final transcripts
    │ Bedrock Claude 3 Haiku analysis
    │ Extract: topic, sentiment, visual concepts
    │ Generate search query
    │ Latency: ~600ms
    ▼
[6] Vector Search (OpenSearch)
    │ KNN search with query embedding
    │ Return top 3 matches
    │ Latency: ~400ms
    ▼
[7] Lambda: Response Handler
    │ Format response
    │ Send via WebSocket to client
    │ Latency: ~100ms
    ▼
[8] Browser: Video Compositor
    │ Preload video from S3/CloudFront
    │ Crossfade transition
    │ Update canvas
    │ Latency: ~500ms (includes network)
    ▼
Total Latency: ~2.5 seconds
```

#### 2.5.2 Optimization Strategies

**Latency Reduction**
1. **Predictive Preloading**: Start loading top 3 clips before final decision
2. **Edge Caching**: CloudFront distribution for video assets
3. **Lambda Warm Pools**: Keep 10 instances warm during peak hours
4. **Batch Embeddings**: Pre-compute embeddings for common phrases

**Cost Optimization**
1. **Tiered Storage**: S3 Intelligent-Tiering for assets
2. **Model Selection**: Use Haiku (fast/cheap) vs Sonnet (accurate/expensive)
3. **Caching Layer**: Redis for frequent queries
4. **Compression**: Serve WebM format (smaller than MP4)

### 2.6 Security Architecture

#### 2.6.1 Authentication Flow

```
User Login
    │
    ▼
[1] Frontend: Cognito Hosted UI
    │ Email/password or social login
    ▼
[2] AWS Cognito User Pool
    │ Validate credentials
    │ Issue JWT tokens (ID, Access, Refresh)
    ▼
[3] Frontend: Store tokens
    │ ID token in memory
    │ Refresh token in httpOnly cookie
    ▼
[4] API Requests
    │ Include ID token in Authorization header
    ▼
[5] API Gateway Authorizer
    │ Validate JWT signature
    │ Check expiration
    │ Extract user claims
    ▼
[6] Lambda Execution
    │ Access userId from event.requestContext
```

#### 2.6.2 Authorization Model

```python
# Role-Based Access Control
PERMISSIONS = {
    'free': {
        'record_minutes': 10,
        'export_quality': '720p',
        'custom_assets': False,
        'api_access': False
    },
    'pro': {
        'record_minutes': -1,  # Unlimited
        'export_quality': '4k',
        'custom_assets': True,
        'api_access': True
    }
}

def check_permission(user_tier: str, action: str) -> bool:
    return PERMISSIONS[user_tier].get(action, False)
```

#### 2.6.3 Data Encryption

- **In Transit**: TLS 1.3 for all connections
- **At Rest**: 
  - S3: AES-256 server-side encryption
  - DynamoDB: AWS managed encryption keys
  - Secrets: AWS Secrets Manager with rotation
- **Client-Side**: Sensitive data never logged or cached

### 2.7 Monitoring & Observability

#### 2.7.1 Metrics Collection

```python
# CloudWatch Custom Metrics
import boto3

cloudwatch = boto3.client('cloudwatch')

def track_latency(stage: str, duration_ms: float):
    cloudwatch.put_metric_data(
        Namespace='LiveDirector',
        MetricData=[{
            'MetricName': f'{stage}_latency',
            'Value': duration_ms,
            'Unit': 'Milliseconds',
            'Dimensions': [
                {'Name': 'Stage', 'Value': stage}
            ]
        }]
    )

# Key metrics to track
METRICS = [
    'transcription_latency',
    'reasoning_latency',
    'retrieval_latency',
    'end_to_end_latency',
    'broll_accuracy',
    'websocket_connections',
    'lambda_cold_starts',
    'error_rate'
]
```

#### 2.7.2 Logging Strategy

```python
import structlog

logger = structlog.get_logger()

# Structured logging for better querying
logger.info(
    "broll_selected",
    session_id=session_id,
    transcript=transcript_text,
    selected_clip=clip_url,
    confidence=score,
    latency_ms=latency
)
```

#### 2.7.3 Alerting Rules

```yaml
Alarms:
  HighLatency:
    Metric: end_to_end_latency
    Threshold: 5000ms
    EvaluationPeriods: 2
    Action: SNS notification to ops team
  
  HighErrorRate:
    Metric: error_rate
    Threshold: 5%
    EvaluationPeriods: 1
    Action: PagerDuty alert
  
  LowAccuracy:
    Metric: broll_accuracy
    Threshold: 70%
    EvaluationPeriods: 3
    Action: Slack notification to ML team
```

### 2.8 Deployment Architecture

#### 2.8.1 Infrastructure as Code

```yaml
# AWS CDK Stack (Python)
from aws_cdk import (
    Stack,
    aws_lambda as lambda_,
    aws_apigatewayv2 as apigw,
    aws_dynamodb as dynamodb,
    aws_s3 as s3,
    aws_cloudfront as cloudfront
)

class LiveDirectorStack(Stack):
    def __init__(self, scope, id):
        super().__init__(scope, id)
        
        # DynamoDB Tables
        sessions_table = dynamodb.Table(
            self, "SessionsTable",
            partition_key=dynamodb.Attribute(
                name="sessionId",
                type=dynamodb.AttributeType.STRING
            ),
            billing_mode=dynamodb.BillingMode.PAY_PER_REQUEST
        )
        
        # S3 Bucket for assets
        assets_bucket = s3.Bucket(
            self, "AssetsBucket",
            encryption=s3.BucketEncryption.S3_MANAGED,
            cors=[s3.CorsRule(
                allowed_methods=[s3.HttpMethods.GET],
                allowed_origins=["https://livedirector.ai"]
            )]
        )
        
        # CloudFront Distribution
        distribution = cloudfront.Distribution(
            self, "CDN",
            default_behavior=cloudfront.BehaviorOptions(
                origin=origins.S3Origin(assets_bucket)
            )
        )
        
        # Lambda Functions
        message_handler = lambda_.Function(
            self, "MessageHandler",
            runtime=lambda_.Runtime.PYTHON_3_11,
            handler="handler.main",
            code=lambda_.Code.from_asset("lambda"),
            timeout=Duration.seconds(30),
            memory_size=1024,
            environment={
                "SESSIONS_TABLE": sessions_table.table_name
            }
        )
        
        # WebSocket API
        websocket_api = apigw.WebSocketApi(
            self, "WebSocketAPI",
            connect_route_options=apigw.WebSocketRouteOptions(
                integration=integrations.WebSocketLambdaIntegration(
                    "ConnectIntegration",
                    connect_handler
                )
            )
        )
```

#### 2.8.2 CI/CD Pipeline

```yaml
# GitHub Actions Workflow
name: Deploy LiveDirector

on:
  push:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Run tests
        run: |
          npm test
          pytest tests/
  
  deploy:
    needs: test
    runs-on: ubuntu-latest
    steps:
      - name: Configure AWS credentials
        uses: aws-actions/configure-aws-credentials@v2
        with:
          role-to-assume: ${{ secrets.AWS_ROLE }}
          aws-region: ap-south-1
      
      - name: Deploy CDK stack
        run: |
          cdk deploy --require-approval never
      
      - name: Deploy frontend
        run: |
          npm run build
          aws s3 sync build/ s3://livedirector-frontend/
          aws cloudfront create-invalidation --distribution-id $DIST_ID --paths "/*"
```


### 2.9 Scalability Design

#### 2.9.1 Horizontal Scaling Strategy

```
Component              | Scaling Method           | Trigger
-----------------------|--------------------------|---------------------------
Lambda Functions       | Auto (AWS managed)       | Request volume
DynamoDB              | On-demand capacity       | Read/write throughput
OpenSearch            | Data nodes + replicas    | Query latency > 500ms
S3                    | Infinite (AWS managed)   | N/A
CloudFront            | Auto (AWS managed)       | Request volume
API Gateway           | Auto (AWS managed)       | Connection count
```

#### 2.9.2 Load Testing Targets

```python
# Locust load test configuration
from locust import HttpUser, task, between

class LiveDirectorUser(HttpUser):
    wait_time = between(1, 3)
    
    @task
    def record_session(self):
        # Simulate 5-minute recording session
        ws = self.client.ws_connect("/ws")
        
        for i in range(300):  # 5 minutes at 1 chunk/sec
            audio_chunk = generate_test_audio()
            ws.send(audio_chunk)
            time.sleep(1)
        
        ws.close()

# Load test scenarios
SCENARIOS = {
    'baseline': {
        'users': 100,
        'spawn_rate': 10,
        'duration': '10m'
    },
    'peak': {
        'users': 1000,
        'spawn_rate': 50,
        'duration': '30m'
    },
    'stress': {
        'users': 5000,
        'spawn_rate': 100,
        'duration': '1h'
    }
}
```

### 2.10 Error Handling & Resilience

#### 2.10.1 Failure Modes & Recovery

```python
class ResilientPipeline:
    def __init__(self):
        self.max_retries = 3
        self.fallback_enabled = True
    
    async def process_audio(self, audio_data):
        try:
            # Primary path: Full AI pipeline
            transcript = await self.transcribe(audio_data)
            context = await self.analyze(transcript)
            clips = await self.search(context)
            return clips[0]
        
        except TranscriptionError as e:
            # Fallback 1: Use cached transcript
            logger.warning("Transcription failed, using cache", error=str(e))
            return self.get_cached_result()
        
        except ReasoningError as e:
            # Fallback 2: Keyword-based search
            logger.warning("AI reasoning failed, using keywords", error=str(e))
            keywords = self.extract_keywords(transcript)
            return await self.keyword_search(keywords)
        
        except RetrievalError as e:
            # Fallback 3: Default B-roll
            logger.error("Search failed, using default", error=str(e))
            return self.get_default_broll()
        
        except Exception as e:
            # Last resort: Continue without overlay
            logger.critical("Pipeline failed completely", error=str(e))
            self.notify_ops_team(e)
            return None
```

#### 2.10.2 Circuit Breaker Pattern

```python
from circuitbreaker import circuit

@circuit(failure_threshold=5, recovery_timeout=60)
async def call_bedrock(prompt: str):
    """
    Circuit breaker prevents cascading failures
    Opens after 5 consecutive failures
    Attempts recovery after 60 seconds
    """
    response = await bedrock_client.invoke_model(
        modelId="anthropic.claude-3-haiku-20240307-v1:0",
        body=json.dumps({"prompt": prompt})
    )
    return response
```

### 2.11 Cost Optimization

#### 2.11.1 Cost Breakdown (per 1000 users/month)

```
Service                | Usage                    | Cost
-----------------------|--------------------------|------------
Lambda                 | 50M requests, 512MB      | $25
API Gateway (WS)       | 10M messages             | $10
Transcribe             | 1000 hours audio         | $1,440
Bedrock (Claude Haiku) | 5M tokens                | $1.25
Bedrock (Titan Embed)  | 10M tokens               | $0.30
OpenSearch             | t3.small.search x2       | $60
S3                     | 1TB storage, 10TB egress | $115
CloudFront             | 10TB data transfer       | $850
DynamoDB               | On-demand, 10M reads     | $12.50
-----------------------|--------------------------|------------
Total                                              | $2,514
Cost per user                                      | $2.51
```

#### 2.11.2 Cost Reduction Strategies

1. **Aggressive Caching**
```python
# Cache frequent queries in Redis
@cache(ttl=3600)
async def search_clips(query: str):
    return await opensearch.search(query)
```

2. **Batch Processing**
```python
# Batch embed multiple assets at once
async def batch_embed_assets(assets: list):
    # Process 25 assets per Bedrock call
    for batch in chunks(assets, 25):
        embeddings = await bedrock.batch_embed(batch)
```

3. **Smart Model Selection**
```python
# Use cheaper models for simple queries
def select_model(complexity: str):
    if complexity == 'simple':
        return 'claude-3-haiku'  # $0.25/1M tokens
    else:
        return 'claude-3-sonnet'  # $3/1M tokens
```

4. **Reserved Capacity**
```python
# Reserve OpenSearch instances for 30% savings
opensearch_reserved_instances = {
    'instance_type': 't3.small.search',
    'instance_count': 2,
    'reservation_term': '1_YEAR'
}
```

### 2.12 Testing Strategy

#### 2.12.1 Unit Tests

```python
# tests/test_reasoning_service.py
import pytest
from services.reasoning import ReasoningService

@pytest.mark.asyncio
async def test_sentiment_detection():
    service = ReasoningService()
    
    # Test positive sentiment
    result = await service.analyze("The stock market is booming!")
    assert result['sentiment'] == 'positive'
    
    # Test negative sentiment
    result = await service.analyze("The economy crashed badly")
    assert result['sentiment'] == 'negative'

@pytest.mark.asyncio
async def test_topic_extraction():
    service = ReasoningService()
    result = await service.analyze("Apple released a new iPhone")
    
    assert 'technology' in result['topic'].lower()
    assert 'apple' in result['visual_concepts']
```

#### 2.12.2 Integration Tests

```python
# tests/test_pipeline.py
import pytest
from pipeline import LiveDirectorPipeline

@pytest.mark.integration
async def test_end_to_end_pipeline():
    pipeline = LiveDirectorPipeline()
    
    # Simulate audio input
    audio_file = "tests/fixtures/sample_speech.wav"
    
    # Run full pipeline
    result = await pipeline.process(audio_file)
    
    # Assertions
    assert result['transcript'] is not None
    assert len(result['clips']) > 0
    assert result['latency_ms'] < 3000
```

#### 2.12.3 Performance Tests

```python
# tests/test_performance.py
import pytest
import time

@pytest.mark.performance
async def test_latency_requirements():
    service = ReasoningService()
    
    start = time.time()
    result = await service.analyze("Test transcript")
    latency = (time.time() - start) * 1000
    
    assert latency < 1000, f"Reasoning took {latency}ms, expected < 1000ms"
```

### 2.13 API Design

#### 2.13.1 REST API Endpoints

```yaml
# Public API (for Pro users)

POST /api/v1/sessions
  Description: Create new recording session
  Auth: Bearer token
  Request:
    {
      "language": "hi-IN",
      "layout": "pip"
    }
  Response:
    {
      "sessionId": "sess_abc123",
      "wsUrl": "wss://api.livedirector.ai/ws?sessionId=sess_abc123"
    }

GET /api/v1/sessions/{sessionId}
  Description: Get session details
  Auth: Bearer token
  Response:
    {
      "sessionId": "sess_abc123",
      "status": "completed",
      "duration": 300,
      "transcript": "...",
      "videoUrl": "https://cdn.livedirector.ai/videos/sess_abc123.mp4"
    }

POST /api/v1/assets/upload
  Description: Upload custom B-roll
  Auth: Bearer token
  Request: multipart/form-data
    - file: video file
    - title: string
    - tags: array
  Response:
    {
      "assetId": "asset_xyz789",
      "url": "https://cdn.livedirector.ai/assets/asset_xyz789.mp4"
    }

GET /api/v1/assets/search
  Description: Search asset library
  Auth: Bearer token
  Query params:
    - q: search query
    - limit: number (default 10)
  Response:
    {
      "results": [
        {
          "assetId": "asset_123",
          "title": "Stock market crash",
          "url": "...",
          "score": 0.95
        }
      ]
    }
```

#### 2.13.2 WebSocket Protocol

```javascript
// Client -> Server messages
{
  "action": "sendAudio",
  "audioData": "base64_encoded_pcm",
  "timestamp": 1234567890
}

{
  "action": "manualOverride",
  "assetId": "asset_123"
}

// Server -> Client messages
{
  "type": "transcription",
  "text": "The stock market",
  "isFinal": false,
  "timestamp": 1234567890
}

{
  "type": "broll_suggestion",
  "clips": [
    {
      "assetId": "asset_123",
      "url": "https://...",
      "confidence": 0.95
    }
  ]
}

{
  "type": "overlay_command",
  "assetId": "asset_123",
  "videoUrl": "https://...",
  "transition": "crossfade"
}

{
  "type": "error",
  "code": "TRANSCRIPTION_FAILED",
  "message": "Unable to process audio"
}
```

### 2.14 Future Enhancements

#### 2.14.1 Phase 2: Generative Video

```python
# Integration with Amazon Nova (when available)
class GenerativeVideoService:
    async def generate_clip(self, prompt: str):
        """
        Generate custom video clip instead of retrieving stock footage
        """
        response = await bedrock.invoke_model(
            modelId="amazon.nova-video-v1",
            body={
                "prompt": prompt,
                "duration": 5,
                "resolution": "1080p"
            }
        )
        return response['videoUrl']
```

#### 2.14.2 Phase 3: Dynamic Chart Generation

```python
# Real-time chart rendering
class ChartGenerator:
    def generate_chart(self, data: dict):
        """
        Generate chart from spoken data
        Example: "Sales grew by 20%" -> Bar chart
        """
        chart_type = self.detect_chart_type(data)
        
        if chart_type == 'bar':
            return self.create_bar_chart(data)
        elif chart_type == 'line':
            return self.create_line_chart(data)
```

## 3. Deployment Environments

### 3.1 Environment Configuration

```yaml
Environments:
  development:
    domain: dev.livedirector.ai
    bedrock_region: us-east-1
    opensearch_size: t3.small.search
    lambda_memory: 512MB
    
  staging:
    domain: staging.livedirector.ai
    bedrock_region: us-east-1
    opensearch_size: t3.medium.search
    lambda_memory: 1024MB
    
  production:
    domain: livedirector.ai
    bedrock_region: ap-south-1
    opensearch_size: r6g.large.search
    lambda_memory: 2048MB
    multi_az: true
    backup_enabled: true
```

## 4. Documentation & Support

### 4.1 Developer Documentation
- API reference (OpenAPI/Swagger)
- WebSocket protocol specification
- SDK examples (Python, JavaScript)
- Integration guides

### 4.2 User Documentation
- Quick start guide
- Video tutorials
- FAQ
- Troubleshooting guide

### 4.3 Operations Runbook
- Deployment procedures
- Rollback procedures
- Incident response playbook
- Monitoring dashboard setup

---

**Document Version**: 1.0  
**Last Updated**: February 15, 2026  
**Team**: StreamLogic  
**Project**: LiveDirector AI
