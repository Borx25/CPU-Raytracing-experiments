#pragma once

#include <variant>
#include "Core.hpp"
#include "Bitmanip.hpp"

template<typename T, size_t N>
struct CyclicBuffer {
	using value_type = T;
	static constexpr size_t Size = N;
	value_type data[Size];

	CyclicBuffer() : end(0), data{} {}

	value_type& operator[](size_t i) noexcept { return data[((end - Size) + i) % Size]; }
	const value_type& operator[](size_t i) const noexcept { return data[((end - Size) + i) % Size]; }

	template<typename Val>
	void push_back(Val&& value) {
		data[end] = value;
		++end %= Size;
	}
	size_t end;
};

template <typename T, size_t Capacity>
struct Stack {
    T elems[Capacity];
    size_t size = 0;

    bool empty() const { return !size; }
    bool full() const { return size >= Capacity; }

	template<typename U> requires(requires(T t, U&& u) {t = u;})
    void push(U&& val) {
        assert(!full());
        elems[size++] = static_cast<U&&>(val);
    }

    T pop() {
        assert(!empty());
        return elems[--size];
    }
};