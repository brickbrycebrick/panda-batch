import asyncio
from limits import RateLimitItemPerMinute
from limits.aio.strategies import MovingWindowRateLimiter
from limits.aio.storage import MemoryStorage


class TokenRateLimiter:
    """Rate limiter for token-based API calls using limits library"""
    
    def __init__(self, max_tokens_per_minute=704000, safety_margin=0.9, max_concurrent=50):
        effective_tokens = max(1, int(max_tokens_per_minute * safety_margin))
        self.token_limit = RateLimitItemPerMinute(effective_tokens)
        self.storage = MemoryStorage()
        self.rate_limiter = MovingWindowRateLimiter(self.storage)
        self.semaphore = asyncio.Semaphore(max_concurrent)
    
    async def acquire(self, estimated_tokens=100):
        """Acquire permission to make an API call with estimated token usage"""
        async with self.semaphore:
            await self.rate_limiter.hit(self.token_limit, "tokens", cost=estimated_tokens)
