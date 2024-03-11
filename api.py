from fastapi import APIRouter, HTTPException, Depends, Response, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import select
from langchain_core.embeddings import Embeddings

from loguru import logger
from typing import List
import json


from schemas import PredictRequest, UserPersonaRequest, UserPersonaResponse
from model import load_model, load_embeddings, make_prediction, make_embeds, make_embed
from database import UserPersona, engine, get_db_session


router = APIRouter()


@router.post("/predict")
async def predict(request: PredictRequest, 
                  tokenizer_and_model: tuple = Depends(load_model), 
                  embeddings: Embeddings = Depends(load_embeddings),
                  db: AsyncSession = Depends(get_db_session)) -> Response:
    tokenizer, model = tokenizer_and_model
    embeddings = embeddings

    predictions = make_prediction(request.user_inputs, tokenizer, model)
    predictions = predictions.split(",")

    embeds = make_embeds(request.user_inputs, embeddings)

    user_personas = [
        UserPersona(username=request.username, persona=prediction, embedding=embed)
        for prediction, embed in zip(predictions, embeds)
    ]

    db.add_all(user_personas)
    await db.commit() # not refresh so that the id will be lately updated

    return PlainTextResponse(f"Persona for {request.username} successfully saved", status_code=201)




@router.get("/persona")
async def get_persona(request: UserPersonaRequest, 
                      embeddings: Embeddings = Depends(load_embeddings),
                      db: AsyncSession = Depends(get_db_session)) -> List[UserPersonaResponse]:
    embeddings = embeddings

    user_input_embed = make_embed(request.user_input, embeddings) 

    # https://github.com/pgvector/pgvector-python/blob/master/README.md#sqlmodel
    # l2_distance, cosine_distance, max_inner_product
    result = await db.execute(
        select(UserPersona)
        .filter(UserPersona.username == request.username)
        .order_by(UserPersona.consine_distance(user_input_embed))
        .limit(request.top_k)
    )
    user_personas = result.scalars().all()

    if not user_personas:
        return PlainTextResponse(f"No persona found for this user {request.username}", status_code=404)
    return [
        UserPersonaResponse(
            persona=user_persona.persona,
        )
        for user_persona in user_personas
    ]



# maintain single connection from generation app
@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket,
                             tokenizer_and_model: tuple = Depends(load_model), 
                             embeddings: Embeddings = Depends(load_embeddings),
                             db: AsyncSession = Depends(get_db_session)):
    tokenizer, model = tokenizer_and_model
    embeddings = embeddings

    await websocket.accept()
    await websocket.send_text("Connection successful")

    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)  # Parse the JSON message

            # from post("/predict")
            if message["type"] == "post_predict":
                # Prepare some summary to send back
                predictions = make_prediction(message["user_inputs"], tokenizer, model)
                predictions = predictions.split(",")

                embeds = make_embeds(message["user_inputs"], embeddings)

                user_personas = [
                    UserPersona(username=message["username"], persona=prediction, embedding=embed)
                    for prediction, embed in zip(predictions, embeds)
                ]

                db.add_all(user_personas)
                await db.commit() # not refresh so that the id will be lately updated
                await websocket.send_text("success")

            # from get("/persona")
            elif message["type"] == "get_persona":
                user_input_embed = make_embed(message["user_input"], embeddings) 
                result = await db.execute(
                    select(UserPersona)
                    .filter(UserPersona.username == message["username"])
                    .order_by(UserPersona.consine_distance(user_input_embed))
                    .limit(message["top_k"])
                )
                user_personas = result.scalars().all()
                user_personas = [{"persona": user_persona.persona} for user_persona in user_personas]
                await websocket.send_text(json.dumps(user_personas))

    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")




