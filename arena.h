#pragma once

#ifdef __cplusplus
extern "C" {
#endif


typedef struct {
    char* start;    // Start of the memory block
    char* current;  // Next available position
    char* end;      // End of the memory block
} Arena;


#define allocate(arena, type, n) \
    ((type*)arena_alloc((arena), (n)*sizeof((type))))

Arena* arena_create(size_t size);
void* arena_alloc(Arena* arena, size_t size);
void arena_free(Arena* arena); 


#ifdef __cplusplus
}
#endif
