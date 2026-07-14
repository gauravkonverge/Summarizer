"""Timeline metric extraction."""

import logging
from datetime import datetime, timezone

from app.models.schemas import Message, TimelineMetrics

logger = logging.getLogger(__name__)


def _normalized_timestamp(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def timeline_metrics(messages: list[Message]) -> TimelineMetrics | None:
    timestamped = [message for message in messages if message.timestamp]
    if len(timestamped) < 2:
        return None
    try:
        timestamps = sorted(_normalized_timestamp(message.timestamp or "") for message in timestamped)
    except (ValueError, TypeError, AttributeError) as exc:
        logger.warning("Timeline timestamps could not be parsed: %s", exc)
        return None
    start, end = timestamps[0], timestamps[-1]
    return TimelineMetrics(
        duration_days=(end - start).days,
        message_count=len(messages),
        start_date=start.isoformat(),
        end_date=end.isoformat(),
        support_message_count=sum(message.role == "support" for message in messages),
        customer_message_count=sum(message.role == "customer" for message in messages),
        unique_senders=len({message.sender for message in messages if message.sender}),
    )
