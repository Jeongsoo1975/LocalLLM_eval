
graph TD
    A[Start: 스크립트 시작] --> B{환경 변수 로드 (.env)}
    B -- XAI_API_KEY 없음 --> BX[오류 출력 후 종료]
    B -- XAI_API_KEY 있음 --> C[xAI 클라이언트 초기화]
    C --> D{xAI API 키 유효성 테스트}
    D -- 유효하지 않음 (401 등) --> DX[오류 출력 후 종료]
    D -- 유효함 --> E{입력 CSV 데이터셋 로드}
    E -- 파일 없음 / 읽기 오류 --> EX[오류 출력 후 종료]
    E -- 로드 성공 --> F[결과 저장용 CSV 파일 준비 (헤더 작성)]
    F --> G{질문 데이터셋 루프 시작 (각 질문에 대해 반복)}
    G -- 다음 질문 있음 --> H[현재 질문 정보 가져오기 (ID, 프롬프트)]
    H --> I[Ollama 응답 요청 함수 호출: get_ollama_response]
    I --> J[Ollama 모델 (gemma3:12b)에 프롬프트 전달]
    J --> K[Ollama로부터 답변 수신]
    K --> L[xAI 평가 요청 함수 호출: get_xai_evaluation]
    L --> M[xAI Grok 모델용 평가 프롬프트 생성 (원본질문 + Ollama답변 + 지침)]
    M --> N[API 요청 전 잠시 대기 (REQUEST_DELAY_SECONDS)]
    N --> O[xAI Grok 모델에 평가 프롬프트 전달]
    O --> P[xAI Grok으로부터 평가 응답 (텍스트) 수신]
    P --> Q[응답 텍스트에서 점수 및 근거 파싱]
    Q --> R[현재 질문에 대한 점수 및 근거 콘솔 출력]
    R -- 평가 성공 (점수 != -1) --> S[총점 및 성공 카운트에 현재 점수 추가]
    R -- 평가 실패/오류 (점수 == -1) --> S
    S --> T[현재 질문의 모든 결과 (질문, Gemma답변, xAI점수, xAI근거)를 CSV에 저장]
    T --> G % 루프의 다음 반복으로 돌아감
    G -- 모든 질문 처리 완료 --> U{최종 평균 점수 계산}
    U -- 성공적으로 평가된 질문 없음 --> UX[평가된 질문 없음 메시지 출력]
    U -- 성공적으로 평가된 질문 있음 --> V[최종 요약 정보 콘솔 출력 (총 질문수, 성공 평가수, 평균점수)]
    V --> W[End: 스크립트 종료 / 결과 파일 저장 완료 메시지]
    UX --> W

    subgraph Ollama Interaction
        J
        K
    end

    subgraph xAI Grok Interaction
        O
        P
        Q
    end